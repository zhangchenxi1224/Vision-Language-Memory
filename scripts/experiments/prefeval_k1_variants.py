"""Reviewed initial-exchange variants; never expose Reader questions to Writer."""
import copy
import json
from pathlib import Path


def load_variants(path, rows):
    artifact=json.loads(Path(path).read_text(encoding='utf-8'))
    assert artifact['review_complete'] is True
    result={}
    for row in rows:
        pid=row['base_pair_id']
        item=artifact['items'][pid]
        assert item['preference']==row['history'][0]['content']
        variants=item['variants']
        assert len(variants)==3 and all(isinstance(v,str) and v.strip() for v in variants)
        assert variants[0]==row['history'][1]['content']
        assert variants[2] not in variants[:2], 'V2 must be held-out wording'
        result[pid]=variants
    return result


def apply_variant(row, variants, index):
    assert index in [0,1,2]
    modified=copy.deepcopy(row)
    modified['history'][1]['content']=variants[row['base_pair_id']][index]
    return modified


def training_variant(cycle, enabled):
    # V2 is NEVER a training condition, even though it shares the review artifact.
    return cycle % 2 if enabled else 0


def select_fm_target(teacher, source, position, source_target):
    return source if source_target and position > 0 else teacher
