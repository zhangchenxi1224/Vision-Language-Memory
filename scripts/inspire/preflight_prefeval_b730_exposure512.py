"""Check the actual B730 checkpoint and unchanged FM loop before continuation."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import torch

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from scripts.experiments.prefeval_k1_data import sha
ROOT=Path('/inspire/ssd/project/exploration-topic/czxs26210936')
RUN=ROOT/'runs/prefeval-b730-exposure512-20260927'
PARENT=ROOT/'runs/prefeval-b-mcq-20260925/robust730/train'
old=subprocess.check_output(['git','show','5d1045f:scripts/experiments/prefeval_k1_writer.py'],cwd=REPO)
current=(REPO/'scripts/experiments/prefeval_k1_write_extension.py').read_bytes()
def loop(source):
    return next(x for f in ast.parse(source).body if isinstance(f,ast.FunctionDef) and f.name=='train'
                for x in f.body if isinstance(x,ast.For) and ast.unparse(x.target)=='step')
assert ast.dump(loop(old))==ast.dump(loop(current))
assert torch.__version__=='2.7.0a0+ecf3bae40a.nv25.02' and torch.version.cuda=='12.8'
assert torch.cuda.device_count()==1  # Preflight and training both expose physical GPU0 only.
assert 'H200' in torch.cuda.get_device_name(0)
assert torch.ones(1,device='cuda').item()==1
print('Loading original complete B730 checkpoint',flush=True)
saved=torch.load(PARENT/'resume.pt',map_location='cpu',weights_only=False)
manifest=json.loads((PARENT/'manifest.json').read_text())
assert saved['manifest']==manifest
assert manifest['implementation_sha256']==hashlib.sha256(old).hexdigest()
assert saved['optimizer_step']==23360 and saved['episode_cursor']==93440
assert len(saved['rng_state']['torch_cuda'])==1
assert len(manifest['targets'])==730
assert len(saved['optimizer']['state'])==1075
assert {int(v['step']) for v in saved['optimizer']['state'].values()}=={23360}
g=saved['optimizer']['param_groups'][0]
assert (g['lr'],g['betas'],g['eps'],g['weight_decay'])==(5e-5,(.9,.999),1e-8,1e-4)
final=torch.load(PARENT/'checkpoint-final.pt',map_location='cpu',weights_only=False)
assert saved['trainable_state'].keys()==final['trainable_state'].keys()
assert all(torch.equal(v,final['trainable_state'][k]) for k,v in saved['trainable_state'].items())
del final,saved
counts=Counter()
records=[json.loads(line) for line in (PARENT/'optimization.jsonl').read_text().splitlines()]
assert [r['step'] for r in records]==list(range(1,23361))
for r in records:
    for draw in r['draws']:
        assert draw['position']==0
        counts[draw['pair_id'],draw['initial_variant']]+=1
assert len(counts)==1460 and set(counts.values())=={64}
RUN.mkdir(parents=True,exist_ok=True)
result=dict(host=socket.gethostname(),torch=torch.__version__,cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0),source_resume_sha256=sha(PARENT/'resume.pt'),
    source_final_sha256=sha(PARENT/'checkpoint-final.pt'),original_writer_sha256=hashlib.sha256(old).hexdigest(),
    current_writer_sha256=hashlib.sha256(current).hexdigest(),optimizer_step=23360,optimizer_states=1075,
    source_final_weights_equal=True,fm_loop_ast_identical=True,
    original_exposures_each_variant=64,final_exposures_each_variant=256,
    fixed_endpoints=[46720,70080,93440],registered_plan_sha256=sha(REPO/'reports/prefeval-k1-l0-l2-20260924/B730_EXPOSURE512_PLAN_20260927.md'))
(RUN/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
