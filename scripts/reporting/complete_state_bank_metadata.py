"""Create a new state-bank manifest with omitted, already verified snapshot bindings."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def complete_bindings(bank,parent,source_sha,parent_sha):
    if bank['models']!=parent['models'] or bank['provenance']['parent_bank_sha256']!=parent_sha:
        raise ValueError('Model bindings do not match the original verified teacher bank')
    snapshots=parent['snapshots']
    if not snapshots or any(not all(v.get(k) for k in ('repo_id','revision','snapshot_payload_sha256')) for v in snapshots.values()):
        raise ValueError('Original bank lacks complete model snapshot identities')
    if 'snapshots' in bank:
        raise ValueError('Metadata completion is only for the recorded missing field')
    result=copy.deepcopy(bank)
    result['snapshots']=copy.deepcopy(snapshots)
    result['metadata_completion']={'source_state_bank_sha256':source_sha,'original_teacher_bank_sha256':parent_sha,
        'changed_field':'snapshots','target_tensors_and_state_groups_unchanged':True}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle-output',type=Path,required=True)
    p.add_argument('--parent-bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    source=a.oracle_output/'bank/manifest.json'
    source_sha=sha(source)
    parent_sha=sha(a.parent_bank)
    if source_sha!='4e68df0fe38a4eb38c6cbf1903229ac81598d7b582c480e1700dbcf69e15b2f9' or parent_sha!='20ef4a9fc53b254fd99b12cbc01cf1a6d41dee8d04dd3120c70ecaa141f30722':
        raise ValueError('This repair is bound to the exact recorded manifests')
    complete=json.loads((a.oracle_output/'complete.json').read_text())
    if not complete['bank_sealed'] or complete['bank_manifest_sha256']!=source_sha:
        raise ValueError('The original oracle bank is not sealed and verified')
    bank=json.loads(source.read_text())
    parent=json.loads(a.parent_bank.read_text())
    for teacher in bank['teachers']:
        if sha(Path(teacher['latent_path']))!=teacher['latent_file_sha256']:
            raise ValueError('A target tensor file changed')
        if sha(Path(teacher['source_run'])/'generations.jsonl')!=teacher['generation_file_sha256']:
            raise ValueError('A target generation record changed')
    result=complete_bindings(bank,parent,source_sha,parent_sha)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,sort_keys=True)
        f.write('\n')
    print(json.dumps({'new_manifest':str(a.output),'sha256':sha(a.output),'source_preserved':sha(source)==source_sha}))


if __name__=='__main__':main()
