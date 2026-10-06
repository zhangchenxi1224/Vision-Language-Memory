import copy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.inspire import run_context_layout as m


def fixture_rows():
    spec=m.read(m.p.PROTOCOL)
    ids=[r['base_pair_id'] for r in m.p.population()[0]]
    old_keys=m.p.expected_keys(ids,spec)|m.new_keys(ids,spec,(m.e.ENDPOINT,))
    def rows(keys):
        output=[]
        for key in sorted(keys):
            pid,qid,endpoint,control,chain=key
            # Canonical specificity 1; new layout worsens matching by .5,
            # specificity drops by .5, without changing teacher prefixes.
            value=2.+(control=='mismatch')
            if endpoint==m.ENDPOINT and control=='memory':value+=.5
            r=dict(zip(m.KEYS,key));r.update(kl=value,teacher_target=pid+qid,
                target_ids=[1,2],teacher_logits_sha256=pid+qid)
            output.append(r)
        return output
    return spec,rows(old_keys),rows(m.new_keys(ids,spec,(m.ENDPOINT,)))


def test_content_and_order_are_exact_including_embedded_role_delimiters():
    exchange=[dict(role='user',content='  中文\nassistant: literal\n'),
        dict(role='assistant',content='### user\nverbatim\t  ')]
    original=copy.deepcopy(exchange)
    assert m.layout_text(exchange)=='### user\n  中文\nassistant: literal\n\n\n### assistant\n### user\nverbatim\t  '
    assert exchange==original
    for row in m.p.population()[0]:
        a,b=row['history'][:2]
        assert m.layout_text([a,b])=='### user\n'+a['content']+'\n\n### assistant\n'+b['content']


@pytest.mark.parametrize('exchange',[[],[dict(role='assistant',content='x'),dict(role='user',content='y')],
    [dict(role='user',content=None),dict(role='assistant',content='x')]])
def test_wrong_exchange_rejected(exchange):
    with pytest.raises(ValueError):m.layout_text(exchange)


def test_complete_denominator_stratification_and_direction():
    spec,old,new=fixture_rows();result=m.summarize(old,new,spec)
    assert (result['combined_rows'],result['new_rows'],result['reused_rows'])==(6912,1536,5376)
    for group in result['strata'].values():
        assert group['independent_n']==16
        for f in group['families'].values():
            assert f['layout_minus_canonical']['mean']==.5
            assert f['canonical_minus_layout_specificity']['mean']==.5
            assert f['endpoints'][m.ENDPOINT]['mismatch_minus_memory']['mean']==.5


@pytest.mark.parametrize('corruption',['missing','duplicate','prefix','foreign','nan'])
def test_invalid_readout_rejected(corruption):
    spec,old,new=fixture_rows()
    if corruption=='missing':new.pop()
    if corruption=='duplicate':new.append(new[0])
    if corruption=='prefix':new[0]=dict(new[0],target_ids=[99])
    if corruption=='foreign':new[0]=dict(new[0],pair_id='dev-illegal')
    if corruption=='nan':new[0]=dict(new[0],kl=float('nan'))
    with pytest.raises(ValueError):m.summarize(old,new,spec)


def test_dev_and_failed_attempts_are_charged_once(tmp_path,monkeypatch):
    monkeypatch.setattr(m.d,'remaining_seconds',lambda _:(0,22000,4000))
    folder=tmp_path/'attempts';folder.mkdir()
    (folder/'failed.json').write_text(json.dumps(dict(started=1,finished=301,exit_code=124)))
    receipts=tmp_path/'receipts';receipts.mkdir()
    (receipts/'duplicate.json').write_text(json.dumps(dict(started=1,finished=301)))
    assert m.remaining_seconds(tmp_path)==(3300,26000,300)
    (folder/'active.json').write_text(json.dumps(dict(started=5)))
    with pytest.raises(ValueError,match='Unsettled'):m.remaining_seconds(tmp_path)
