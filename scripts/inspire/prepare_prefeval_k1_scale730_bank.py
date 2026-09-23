"""Reuse the frozen pilot targets and leave the remaining 666 states to optimization."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments.prefeval_k1_data import load_records, load_training_records, sha


def main(args):
    original=json.loads(subprocess.check_output(['git','show',
        '5153444:reports/prefeval-k1-l0-l2-20260924/question-forms.json'],cwd=ROOT,text=True))
    full={r['base_pair_id']:r for r in load_records('train')}
    pilot=load_records('pilot')
    reused=[]
    for arm in ['A','B']:
        destination=args.output/arm
        destination.mkdir(parents=True,exist_ok=True)
        for row in pilot:
            pid=row['base_pair_id']
            assert all(full[pid]['forms'][family]==original[pid][family] for family in ['T1','T2','T3'])
            source=(args.pilot/arm/pid.replace(':','_')).resolve()
            done=json.loads((source/'complete.json').read_text(encoding='utf-8'))
            assert done['step']==288 and done['binding']['arm']==arm
            assert done['latent_sha256']==sha(source/'latent.pt') and done['png_sha256']==sha(source/'memory.png')
            link=destination/source.name
            if link.exists() or link.is_symlink():
                assert link.is_symlink() and link.resolve()==source
            else:
                link.symlink_to(source,target_is_directory=True)
            reused.append({'arm':arm,'pair_id':pid,'source':str(source),
                'latent_sha256':done['latent_sha256'],'png_sha256':done['png_sha256'],
                'original_commit':done['binding']['commit']})
    value={'reused_targets':reused,'new_pair_ids':[r['base_pair_id'] for r in load_training_records('train',True)],
        'per_arm_total':730,'per_arm_reused':64,'per_arm_new':666,
        'training_forms_preserved':True,'reuse_does_not_add_gradient_steps':True}
    path=args.output/'pilot-reuse.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==value
    else:
        path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'per_arm_reused':64,'per_arm_new':666,'new_updates_per_arm':666*288}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
