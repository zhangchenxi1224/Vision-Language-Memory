"""Hash-bound prospective recovery allocation; historical gate stays failed."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.eval.prefeval_rgb import load_overlay
from vision_memory.prefeval.rgb_protocol import digest

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_policy(path,manifest,overlay):
    p=json.loads(Path(path).read_text(encoding='utf-8'))
    root=Path(path).parent
    for name,expected in p['evidence_file_hashes'].items():
        if sha(root/name)!=expected:raise ValueError('Allocation evidence changed: '+name)
    for name,expected in p['evidence_json_digests'].items():
        if digest(json.loads((root/name).read_text(encoding='utf-8')))!=expected:
            raise ValueError('Allocation JSON evidence changed: '+name)
    if p['manifest_digest']!=digest(manifest) or p['overlay_digest']!=digest(overlay):
        raise ValueError('Allocation binding mismatch')
    r=json.loads((root/'references-v2-verified.json').read_text(encoding='utf-8'))
    if not r['verified'] or r['reads']!=3088 or r['gate_passed'] is not False:
        raise ValueError('Historical reference coverage/gate changed')
    if r['mcq'][1]!=96 or r['mcq'][0]<77 or r['recovery'][1]!=928 or r['recovery'][0]<836:
        raise ValueError('Primary readiness failed')
    correct,total=r['supplement_families']['text/family/training_recovery']
    if total!=264 or correct/total<.9:raise ValueError('Training recovery readiness failed')
    if p['sentinel_ids']!=manifest['sentinel_targets']:raise ValueError('Sentinel membership changed')
    return p

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--verification',type=Path,required=True);a=p.parse_args()
    root=a.policy.parent;m=json.loads((root/'registered/manifest.json').read_text(encoding='utf-8'))
    o=load_overlay(root/'reader-format-v2.json',m);policy=load_policy(a.policy,m,o)
    original=json.loads((root/'references-v2-verified.json').read_text(encoding='utf-8'))
    live=json.loads(a.verification.read_text(encoding='utf-8'))
    for field in ('verified','reads','counts','mcq','recovery','supplement_families','gate_passed'):
        if original[field]!=live[field]:raise ValueError('Raw reference reconstruction changed: '+field)
    # The same archive was reconstructed on Windows and Linux. Paths are labels;
    # compare identical relative names and exact file-byte digests across hosts.
    original_files={k.replace('\\','/'):v for k,v in original['files'].items()}
    live_files={k.replace('\\','/'):v for k,v in live['files'].items()}
    if original_files!=live_files:raise ValueError('Raw reference file bytes changed')
    print(json.dumps(dict(allocation_ready=True,historical_combined_gate_passed=False,
                         policy_digest=digest(policy),baseline_writes=124,sentinel_states=40)))
