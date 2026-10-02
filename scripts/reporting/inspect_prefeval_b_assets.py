"""Validate only the task's shared assets and emit a compact deployment receipt."""
import collections
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments.prefeval_k1_data import load_records,sha

P=Path('/inspire/ssd/project/exploration-topic/czxs26210936')
R=P/'runs/prefeval-k1-l0-l2-20260924'
S=P/'runs/prefeval-k1-scale730-20260924'
C=P/'runs/prefeval-k1-initial-source-retain-20260924/B'
result={'teachers':{},'control':{},'other_outputs':{}}
for split,bank in [('pilot',R/'pilot/B'),('train',S/'teachers/B')]:
    rows=load_records(split)
    hashes={}
    for row in rows:
        d=bank/row['base_pair_id'].replace(':','_')
        x=json.loads((d/'complete.json').read_text())
        assert x['step']==288 and x['binding']['arm']=='B'
        assert sha(d/'latent.pt')==x['latent_sha256'] and sha(d/'memory.png')==x['png_sha256']
        hashes[row['base_pair_id']]=x['latent_sha256']
    result['teachers'][split]={'count':len(hashes),'target_hashes':hashes}
manifest=json.loads((C/'train/manifest.json').read_text())
done=json.loads((C/'train/complete.json').read_text())
assert manifest['arm']=='B' and manifest['stage']=='retain' and manifest['steps']==2048
assert manifest['retain_source_mode']=='initial_student_png_for_all_distractor_positions'
assert manifest['targets']==result['teachers']['pilot']['target_hashes']
assert manifest['parent_sha256']==sha(R/'writer/B/write/checkpoint-final.pt')
assert done['checkpoint_sha256']==sha(C/'train/checkpoint-final.pt')
sources={}
for row in load_records('pilot'):
    d=R/'writer/B/training-prefixes'/row['base_pair_id'].replace(':','_')/'seed-0'
    x=json.loads((d/'complete.json').read_text())
    h=sha(d/'prefix-00.png')
    assert x['png_hashes']['prefix-00.png']==h
    sources[row['base_pair_id']]=h
result['control']={'path':str(C),'manifest':manifest,'complete':done,'source_hashes':sources}
for name,path in [('old730',S/'writer/B/write'),('fm8192',P/'runs/prefeval-k1-fm8192-20260924/B/train')]:
    result['other_outputs'][name]={'path':str(path),'exists':path.exists(),
        'complete':json.loads((path/'complete.json').read_text()) if (path/'complete.json').exists() else None}
for split,n in [('pilot',128),('dev',180)]:
    chains=list((C/split).glob('*/seed-*/complete.json'))
    result['control'][split]={'complete_chains':len(chains),'expected':n,
        'readback_finished':(C/f'readback-{split}-T1/finished-0.json').exists()}
out=P/'runs/prefeval-b-mcq-20260925'
out.mkdir(parents=True,exist_ok=True)
(out/'reused-assets.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'teachers':{k:v['count'] for k,v in result['teachers'].items()},
 'control':{k:v for k,v in result['control'].items() if k not in ['manifest','source_hashes']},
 'other_outputs':result['other_outputs']}))
