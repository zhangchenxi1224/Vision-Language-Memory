from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiments.prefeval_context_readout import validate_protocol,expected_keys
from scripts.reporting.context_readout_report import summarize

SPEC=json.loads((ROOT/'configs/experiments/context_readout_audit.json').read_text())

def test_complete_protocol_and_denominator():
    validate_protocol(SPEC)
    assert len(expected_keys([str(i) for i in range(16)],SPEC['queries']))==2496

def test_duplicates_and_selection_leakage_are_rejected():
    bad=deepcopy(SPEC);bad['queries'][1]=bad['queries'][0]
    with pytest.raises(ValueError):validate_protocol(bad)
    from scripts.experiments.prefeval_context_coverage import VALIDATION
    bad=deepcopy(SPEC);bad['queries'][0]['query']=VALIDATION[0][1]
    with pytest.raises(ValueError):validate_protocol(bad)

def test_report_requires_all_rows():
    with pytest.raises(ValueError,match='denominator'):summarize([],['one'],SPEC)

def test_report_averages_by_history_and_uses_lower_kl_as_better():
    rows=[]
    for pid,q,e,c in sorted(expected_keys(['one','two'],SPEC['queries'])):
        kl=4. if c in ('blank','mismatch') else (3. if e.startswith('original') else 1.)
        rows.append({'pair_id':pid,'query_id':q,'endpoint':e,'control':c,'kl':kl})
    result=summarize(rows,['one','two'],SPEC)
    v=result['contrasts']['recall/prompt_matching/coverage_kl_reduction']
    assert v['mean']==2. and v['independent_n']==2 and v['ci95']==[2.,2.]
    with pytest.raises(ValueError,match='Duplicate'):summarize(rows+[rows[0]],['one','two'],SPEC)
