"""Read-only training-bank geometry and FM residuals; never use dev/OOD scores."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch


def diagnose(root):
    result={'source':str(root),'arms':{},'scope':'All64 training teachers only; no generated-student performance claim',
        'identity':'For a training bridge x=(1-sigma)z+sigma*epsilon, z_hat=x-sigma*v_hat; clean MSE=sigma^2*velocity MSE.',
        'limits':'Residuals use target-noised training inputs, not free-running Gaussian trajectories. Cross-window draws differ. Bank-mean error is a geometric reference, not a tested unconditional model.'}
    torch.set_num_threads(2)
    for arm in ('A','B'):
        paths=sorted((root/'teachers'/arm).glob('*/latent.pt'))
        if len(paths)!=64:raise RuntimeError('Require the full fixed64 training bank')
        bank=torch.stack([torch.load(p,map_location='cpu',weights_only=True).float() for p in paths])
        center=bank.mean(dim=0)
        log=root/'writers'/arm/'write/optimization.jsonl'
        rows=[json.loads(line) for line in log.read_text().splitlines()]
        windows={}
        for label,subset in (('first256',rows[:256]),('last256',rows[-256:])):
            bins=[]
            for low in (0.,.25,.5,.75):
                draws=[d for row in subset for d in row['draws'] if low<=d['sigma']<low+.25]
                bins.append(dict(sigma_range=[low,low+.25],draws=len(draws),
                    velocity_mse=statistics.mean(d['mse'] for d in draws),
                    target_noised_clean_mse=statistics.mean(d['sigma']**2*d['mse'] for d in draws)))
            windows[label]=bins
        result['arms'][arm]=dict(states=len(paths),latent_shape=list(bank.shape[1:]),
            bank_mean_mse=float((bank-center).square().mean()),latent_global_rms=float(bank.square().mean().sqrt()),
            fm_steps=len(rows),fm_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),windows=windows)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args();value=diagnose(args.run)
    text=json.dumps(value,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    else:print(text)
