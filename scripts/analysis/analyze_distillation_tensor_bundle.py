"""Zero-update, CPU-only decomposition of saved reference/student tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from vision_memory.training.empirical_bank_flow import EmpiricalBankFlow


def mse(x): return float(x.double().square().mean())
def rms(x): return mse(x)**.5


def spectrum_fit(pred,target):
    # Compare against the per-channel spatial constant predictor. A band has
    # many more bins than another; raw error energy alone is not a fit score.
    pred=pred.double();target=target.double()
    p=torch.fft.fft2(pred,norm='ortho')
    t=torch.fft.fft2(target,norm='ortho')
    h,w=target.shape[-2:]
    radius=torch.maximum(torch.fft.fftfreq(h).abs()[:,None],torch.fft.fftfreq(w).abs()[None,:])
    error=(p-t).abs().square()
    signal=t.abs().square()
    result={}
    for name,mask in [('dc',radius==0),('low',(radius>0)&(radius<=.0625)),('mid',(radius>.0625)&(radius<=.25)),('high',radius>.25)]:
        e=float(error[...,mask].sum());s=float(signal[...,mask].sum())
        result[name]=dict(error_energy_fraction=e/max(float(error.sum()),1e-30),error_to_target_energy=e/max(s,1e-30),
            target_energy_fraction=s/float(signal.sum()),output_to_target_energy=float(p.abs().square()[...,mask].sum())/max(s,1e-30))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    with zipfile.ZipFile(args.bundle) as z:
        meta=json.loads(z.read('metadata.json'))
        cache={}
        def tensor(name):
            spec=meta['tensors'][name];member=spec['archive_storage']
            if member not in cache:
                raw=z.read(member)
                assert hashlib.sha256(raw).hexdigest()==Path(member).stem
                cache[member]=torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy())
            return torch.as_strided(cache[member],tuple(spec['shape']),tuple(spec['stride']),storage_offset=spec['offset']).clone()
        ids=sorted(k.split('/',1)[1] for k in meta['tensors'] if k.startswith('teacher/'))
        teachers=torch.stack([tensor('teacher/'+i) for i in ids])
        source=tensor('source')
        weights=torch.tensor([meta['json']['distill/result']['target_visits'].get(i,0) for i in ids],dtype=torch.float64)
        weights/=weights.sum()
        weighted_mean=(teachers.double()*weights.reshape(-1,1,1,1,1)).sum(0)
        uniform_mean=teachers.double().mean(0)
        report=dict(bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest(),bank_sha256=meta['bank_sha256'],
            optimizer_updates=0,teacher_count=len(ids),bank_dispersion_rms=rms(teachers-uniform_mean),
            bank_to_source_rms=rms(teachers-source),uniform_vs_weighted_mean_rms=rms(uniform_mean-weighted_mean),phases={},single={},trajectories={})
        reference=torch.stack([tensor(f'reference/{i}/latent') for i in range(8)])
        starts=torch.stack([tensor(f'reference/{i}/trajectory/0') for i in range(8)])
        reference_path=[torch.stack([tensor(f'reference/{i}/trajectory/{j}') for i in range(8)]) for j in range(1,5)]
        report['constant_endpoint_controls']={}
        for label,endpoint in [('source',source),('uniform_teacher_mean',uniform_mean),('training_visit_weighted_mean',weighted_mean)]:
            # A zero-training affine path from the identical starts to a fixed
            # endpoint. Means are coordinate controls, not verified positives.
            path=[endpoint+(s/.4999999701976776)*(starts-endpoint) for s in [.375,.25,.1249999925494194,0.]]
            errors=[mse(x-y) for x,y in zip(path,reference_path)]
            report['constant_endpoint_controls'][label]=dict(endpoint_mse=errors[-1],
                intermediate_mse=float(np.mean(errors[:3])),total_distillation_loss=errors[-1]+float(np.mean(errors[:3])))
        for phase in ['reference','baseline','step-000064','step-000256','trained']:
            outputs=torch.stack([tensor(f'{phase}/{i}/latent') for i in range(8)])
            paired=outputs-reference
            centroid_error=outputs.double().mean(0)-reference.double().mean(0)
            centered_error=(outputs.double()-outputs.double().mean(0))-(reference.double()-reference.double().mean(0))
            errors=mse(paired)
            bias=mse(centroid_error);scatter=mse(centered_error)
            assert abs(errors-bias-scatter)<1e-7
            report['phases'][phase]=dict(paired_endpoint_mse=errors,centroid_error_mse=bias,
                centered_error_mse=scatter,output_variance=mse(outputs-outputs.mean(0)),
                reference_variance=mse(reference-reference.mean(0)),
                distance_to_training_target_mean_rms=rms(outputs-weighted_mean),distance_to_source_rms=rms(outputs-source),
                distance_to_uniform_bank_mean_rms=rms(outputs-uniform_mean),
                spectrum_vs_paired_reference=spectrum_fit(outputs,reference))
        for j in range(5):
            target=torch.stack([tensor(f'reference/{i}/trajectory/{j}') for i in range(8)])
            row={}
            for phase in ['baseline','trained']:
                pred=torch.stack([tensor(f'{phase}/{i}/trajectory/{j}') for i in range(8)])
                row[phase]=dict(paired_mse=mse(pred-target),spread_rms=rms(pred-pred.mean(0)),reference_spread_rms=rms(target-target.mean(0)))
                if j==0: assert torch.equal(pred,target)
            report['trajectories'][str(j)]=row
        for phase in ['baseline','trained']:
            errors=[report['trajectories'][str(j)][phase]['paired_mse'] for j in range(1,5)]
            report['phases'][phase]['paired_total_distillation_loss']=errors[-1]+float(np.mean(errors[:3]))
        flow=EmpiricalBankFlow(source,teachers,start_sigma=.4999999701976776)
        for j,sigma in enumerate([.4999999701976776,.375,.25,.1249999925494194]):
            entropies=[];max_probs=[]
            for i in range(8):
                _,prob,_=flow.evaluate(tensor(f'reference/{i}/trajectory/{j}'),sigma)
                entropies.append(float(-(prob*prob.clamp_min(1e-300).log()).sum()))
                max_probs.append(float(prob.max()))
            report['trajectories'][str(j)]['reference_posterior']=dict(entropy_mean=float(np.mean(entropies)),max_probability_mean=float(np.mean(max_probs)))
        for rank,step in [(4,512),(4,2048),(16,512)]:
            identity=meta['json'][f'single/rank{rank}/identity'];anchor=tensor('teacher/'+identity['anchor_id'])
            pred=tensor(f'single/rank{rank}/step{step}/train')
            base=meta['json'][f'single/rank{rank}/baseline-rms']
            case=meta['json'][f'single/rank{rank}/cases']['cases'][0]
            observed=rms(pred-anchor)
            result=meta['json'][f'single/rank{rank}/step{step}/result']
            assert abs(observed-result['train']['mean_latent_rms'])<1e-8
            report['single'][f'rank{rank}-step{step}']=dict(train_rms=observed,
                baseline_rms=base[str(case['noise_seed'])],train_em=result['train']['exact_match'],
                test_em=result['test']['exact_match'],spectrum=spectrum_fit(pred,anchor),
                target_spatial_rms=rms(anchor-anchor.mean((-2,-1),keepdim=True)),
                output_spatial_rms=rms(pred-pred.mean((-2,-1),keepdim=True)))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report))


if __name__=='__main__': main()
