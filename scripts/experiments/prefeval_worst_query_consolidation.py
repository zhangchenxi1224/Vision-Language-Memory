"""Plan12: matched complete-bank mean versus mean/worst-query consolidation."""
from __future__ import annotations
import argparse,hashlib,inspect,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from PIL import Image
from scripts.experiments import prefeval_compositional_evidence as e
q=e.q;s=e.s;j=e.j;digest=e.digest;tensor_sha=e.tensor_sha
DATA=s.REPORT/'worst-query-consolidation-v1'
INSTANCE='dl-clear-retain-h200x4-20260914'


def schedule(step):
    c=step%4;b=(step//4)%2;r=(step//8)%4
    return dict(step=step,case=c,bank=b,rotation=r,format='full-action' if (c+b+r)%2==0 else 'xml')


def recovery_bank(target):
    weights={item['scope']:w for item,w in s.training_jobs(target,dict(recovery_form=0,mixture_application=False),'A')}
    bank=[]
    for scope,value in sorted(target['state'].items()):
        items=[x for x in target['recovery'] if x['scope']==scope]
        assert len(items)==3
        items += [q.recovery_form(scope,value,kind,ix) for kind in ('question','instruction') for ix in range(4)]
        bank.extend((x,weights[scope]/11) for x in items)
    return sorted(bank,key=lambda row:(row[0]['scope'],row[0]['id']))


def applications(target,step,cases):
    spec=schedule(step)
    weights={item['scope']:w for item,w in s.training_jobs(target,dict(recovery_form=0,mixture_application=False),'A')}
    result=[]
    for scope,value in sorted(target['state'].items()):
        if value is None:continue
        old=next(x for x in target['applications'] if x['scope']==scope and x['scenario']==spec['case'])
        case=cases[old['value_id']][spec['case']] if spec['bank'] else old
        item,choices,gold=e.application(case,spec['format'],spec['rotation'])
        result.append((item,.25*weights[scope],choices,gold))
    return result


class RecoveryAccumulator:
    """First-order buffers; no Reader graph survives its own query gradient."""
    def __init__(self,like):
        self.mean_gradient=torch.zeros_like(like)
        self.mean_loss=0.;self.worst_loss=-math.inf;self.worst_identity=None;self.worst_gradient=None

    def add(self,identity,loss,weight,gradient):
        assert math.isfinite(loss) and torch.isfinite(gradient).all()
        self.mean_gradient.add_(gradient,alpha=weight)
        self.mean_loss+=weight*loss
        if loss>self.worst_loss or (loss==self.worst_loss and (self.worst_identity is None or identity<self.worst_identity)):
            self.worst_loss=loss;self.worst_identity=identity;self.worst_gradient=gradient.detach().clone()

    def finish(self,arm):
        assert arm in ('M','W') and self.worst_gradient is not None
        if arm=='M':return self.mean_gradient,self.mean_loss
        return .5*self.mean_gradient+.5*self.worst_gradient,.5*self.mean_loss+.5*self.worst_loss


def functions():
    return {name:hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() for name,fn in
        [('schedule',schedule),('recovery_bank',recovery_bank),('applications',applications),
         ('accumulator_add',RecoveryAccumulator.add),('accumulator_finish',RecoveryAccumulator.finish)]}


def load():
    reg=s.load_json(DATA/'registration.json');parent,p,cases=e.load()
    assert digest(parent)==reg['parent_registration_digest']=='e050b7bd9d807591dbb54987d8f58020af68eb9514e4dd7cdc040ecd6fa1b0ac'
    assert functions()==reg['functions']
    assert digest({sid:recovery_bank(t) for sid,t in p['targets'].items()})==reg['recovery_bank_digest']
    return reg,p,cases


def register(a):
    parent,p,cases=e.load();runtime=s.load_json(a.runtime_receipt)
    assert runtime['instance']==INSTANCE and len(runtime['actual']['gpus'])==4
    assert INSTANCE in runtime['platform_status'] and 'RUNNING' in runtime['platform_status']
    endpoints={sid:s.load_json(a.source/'training/E'/sid/'complete.json') for sid in p['targets']}
    for sid,done in endpoints.items():
        for name,h in done['artifacts'].items():assert s.file_sha(a.source/'training/E'/sid/name)==h
    bank={sid:recovery_bank(t) for sid,t in p['targets'].items()}
    assert sum(map(len,bank.values()))==968
    reg=dict(plan='prefeval-rgb-worst-query-consolidation-12',parent_registration_digest=digest(parent),
        parent_verified=s.file_sha(a.source/'final-verified.json'),instance=INSTANCE,runtime=runtime,
        training_digest=digest(p),evaluation_digest=parent['evaluation_digest'],contrast_values=parent['contrast_values'],
        cases_digest=digest(cases),recovery_bank_digest=digest(bank),functions=functions(),
        implementation={**parent['implementation'],str(Path(__file__).relative_to(ROOT)).replace('\\','/'):s.file_sha(Path(__file__))},
        endpoint_receipts=endpoints,endpoint_receipt_hashes={sid:s.file_sha(a.source/'training/E'/sid/'complete.json') for sid in p['targets']},
        arms=['M','W'],steps=32,inherited_updates=[256,128,64,64,64],optimizer=parent['optimizer'],
        schedule=[schedule(t) for t in range(32)],progression=parent['progression'],
        objectives={'M':'weighted mean recovery + .25 slot-weighted application','W':'.5 weighted mean recovery + .5 unweighted worst query + .25 slot-weighted application'},
        tie_break='lexicographically smallest (scope, query id) among exactly tied current losses',
        budget=dict(updates=2560,gradient_forwards={'M':41728,'W':41728,'total':83456},recovery_forwards=61952,
            application_forwards=21504,generations=5320,ranking_decisions=2256,max_candidate_forwards=9024,
            endpoint_recovery_forwards=2288),writer_updates=0,
        limitation='Observed development endpoints; independent latent teachers, not recurrent Writer transitions. Prior E-R macro gate remains failed.')
    s.write_frozen(DATA/'recovery-bank.json',bank);s.write_frozen(DATA/'registration.json',reg)
    print(json.dumps(dict(registration_digest=digest(reg),budget=reg['budget'])))


def guard(a,reg):
    assert j.actual_host()==reg['runtime']['actual']
    assert s.file_sha(a.source/'final-verified.json')==reg['parent_verified']
    for name,sha in reg['implementation'].items():assert s.file_sha(ROOT/name)==sha


def train(a):
    reg,p,cases=load();guard(a,reg);processor,reader,vae,versions,bindings=s.models(a,True)
    termination=s.assistant_termination_contract(reader,processor);assignment=sorted(p['targets'])[a.shard::a.shards]
    s.write_frozen(a.output/f'train-identity-{a.shard}.json',dict(**s.identity(a,bindings,reg,assignment),actual_host=j.actual_host(),
        prompt_frame=s.query_prompt(processor,'__REGISTERED_QUERY__'),termination=termination))
    for sid in assignment:
        target=p['targets'][sid];src=a.source/'training/E'/sid
        assert s.file_sha(src/'complete.json')==reg['endpoint_receipt_hashes'][sid]
        z=torch.load(src/'latent.pt',map_location=a.device,weights_only=True)
        assert tensor_sha(z)==reg['endpoint_receipts'][sid]['final_tensor_sha']
        for arm in (('M','W') if sorted(p['targets']).index(sid)%2==0 else ('W','M')):
            out=a.output/'training'/arm/sid;out.mkdir(parents=True,exist_ok=False)
            oracle=s.VAELatentOracle(vae=vae,initial_latent=z,compute_dtype=torch.float32)
            opt=torch.optim.Adam([oracle.latent_fp32],**reg['optimizer']);torch.save(z.detach().cpu(),out/'initial-latent.pt')
            started=time.monotonic();forwards=tokens=0
            for step in range(32):
                opt.zero_grad(set_to_none=True);pixels=oracle.image();before=oracle.latent_fp32.detach().clone()
                acc=RecoveryAccumulator(oracle.latent_fp32);losses=[];apps=applications(target,step,cases);bank=recovery_bank(target)
                for ix,(item,w) in enumerate(bank):
                    c0,t0=processor.total_calls,processor.total_input_tokens;processor.begin_capture()
                    ce=s.qwen3vl_answer_eos_ce(model=reader,processor=processor,image=pixels[0],query=item['query'],target=item['target'],device=a.device,
                        termination=termination,lambda_eos=1.,require_image_grad=True,reader_resize_contract=s.R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
                    proof=processor.end_capture();assert processor.total_calls-c0==1 and len(proof)==1
                    gradient=torch.autograd.grad(ce.loss,oracle.latent_fp32,retain_graph=bool(apps) or ix+1<len(bank))[0]
                    identity=(item['scope'],item['id']);value=float(ce.loss.detach());acc.add(identity,value,w,gradient)
                    used=processor.total_input_tokens-t0;forwards+=1;tokens+=used
                    losses.append(dict(kind='recovery',query=item,mean_weight=w,loss=value,answer_ce=float(ce.answer_loss.detach()),
                        eos_ce=float(ce.eos_loss.detach()),answer_tokens=ce.answer_token_count,reader_forwards=1,processed_input_tokens=used,
                        processor_input=proof[0],gradient_norm=float(gradient.norm())))
                    del ce,gradient
                total_gradient,recovery_objective=acc.finish(arm);application_objective=0.
                for ix,(item,w,choices,gold) in enumerate(apps):
                    c0,t0=processor.total_calls,processor.total_input_tokens;processor.begin_capture()
                    ce=s.qwen3vl_listwise_choice_ce(model=reader,processor=processor,image=pixels[0],query=item['query'],choices=choices,target_index=gold,
                        device=a.device,require_image_grad=True,reader_resize_contract=s.R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
                    proof=processor.end_capture();assert processor.total_calls-c0==4
                    gradient=torch.autograd.grad(ce.loss,oracle.latent_fp32,retain_graph=ix+1<len(apps))[0]
                    assert torch.isfinite(gradient).all();total_gradient.add_(gradient,alpha=w)
                    value=float(ce.loss.detach());application_objective+=w*value;used=processor.total_input_tokens-t0;forwards+=4;tokens+=used
                    losses.append(dict(kind='application',query=item,weight=w,loss=value,choices=choices,gold_index=gold,
                        scores=ce.choice_logits.detach().tolist(),candidate_token_counts=ce.choice_token_counts,
                        reader_forwards=4,processed_input_tokens=used,processor_inputs=proof,
                        gradient_norm=float(gradient.norm())))
                    del ce,gradient
                for loss in losses:
                    if loss['kind']=='recovery':
                        is_worst=(loss['query']['scope'],loss['query']['id'])==acc.worst_identity
                        loss['effective_weight']=loss['mean_weight'] if arm=='M' else .5*loss['mean_weight']+.5*int(is_worst)
                assert torch.isfinite(total_gradient).all();oracle.latent_fp32.grad=total_gradient;opt.step()
                assert torch.isfinite(oracle.latent_fp32).all()
                s.append(out/'optimization.jsonl',dict(step=step,schedule=schedule(step),losses=losses,
                    mean_recovery_loss=acc.mean_loss,worst_recovery_loss=acc.worst_loss,worst_query=acc.worst_identity,
                    recovery_objective=recovery_objective,application_objective=application_objective,
                    weighted_objective=recovery_objective+application_objective,
                    gradient=dict(norm=float(total_gradient.norm()),displacement_norm=float((oracle.latent_fp32.detach()-before).norm()))))
                if (step+1)%16==0:torch.save(dict(next_step=step+1,latent=oracle.latent_fp32.detach().cpu(),optimizer=opt.state_dict(),registration_digest=digest(reg)),out/'checkpoint.pt')
            torch.save(oracle.latent_fp32.detach().cpu(),out/'latent.pt');torch.save(opt.state_dict(),out/'optimizer.pt')
            with torch.no_grad():array=oracle.image()[0].detach().cpu().mul(255).round().clamp(0,255).byte().permute(1,2,0).numpy()
            Image.fromarray(array).save(out/'memory.png');s.frozen(versions)
            s.write_frozen(out/'complete.json',dict(target=sid,arm=arm,inherited_updates=reg['inherited_updates'],additional_updates=32,
                initial_tensor_sha=tensor_sha(z),final_tensor_sha=tensor_sha(oracle.latent_fp32),reader_forwards=forwards,processed_input_tokens=tokens,
                seconds=time.monotonic()-started,artifacts={x:s.file_sha(out/x) for x in ('initial-latent.pt','latent.pt','optimizer.pt','checkpoint.pt','optimization.jsonl','memory.png')}))
            del opt,oracle
    s.write_frozen(a.output/f'train-complete-{a.shard}.json',dict(targets=assignment))


def measure_ce(a,reader,processor,image,item,termination):
    processor.begin_capture()
    with torch.no_grad():
        ce=s.qwen3vl_answer_eos_ce(model=reader,processor=processor,image=image,query=item['query'],target=item['target'],device=a.device,
            termination=termination,lambda_eos=1.,require_image_grad=False,reader_resize_contract=s.R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
    proof=processor.end_capture();vals=ce.target_logits.float();gold=vals.gather(-1,ce.target_ids.unsqueeze(-1)).squeeze(-1)
    nll=torch.logsumexp(vals,-1)-gold;other=vals.clone();other.scatter_(-1,ce.target_ids.unsqueeze(-1),float('-inf'))
    return dict(answer_ce=float(ce.answer_loss),eos_ce=float(ce.eos_loss),loss=float(ce.loss),answer_tokens=ce.answer_token_count,
        target_ids=ce.target_ids[0].tolist(),token_nll=nll[0].tolist(),gold_token_margins=(gold-other.amax(-1))[0].tolist(),
        teacher_forced_correct=(vals.argmax(-1)==ce.target_ids)[0].tolist(),processor_input=proof[0])


def evaluate(a):
    reg,p,new_cases=load();guard(a,reg);e=s.load_json(s.DATA/'evaluation-payload.json');old_cases={}
    for fn in ('training-scenarios.json','reserved-scenarios.json'):
        for cc in s.load_json(s.DATA/fn).values():
            for c in cc:old_cases[(c['id'],c['scope'],c['value_id'])]=c
    endpoints={f'{arm}/{sid}':s.file_sha(a.output/'training'/arm/sid/'complete.json') for sid in p['targets'] for arm in ('M','W')}
    processor,reader,_,versions,bindings=s.models(a);termination=s.assistant_termination_contract(reader,processor)
    assignment=sorted(p['targets'])[a.shard::a.shards];out=a.output/'evaluation'/f'shard-{a.shard}';out.mkdir(parents=True,exist_ok=False)
    frame=s.query_prompt(processor,'__REGISTERED_QUERY__');s.write_frozen(out/'identity.json',dict(**s.identity(a,bindings,reg,assignment),actual_host=j.actual_host(),frozen_endpoints=endpoints,prompt_frame=frame,termination=termination))
    started=time.monotonic();n=rn=cf=cen=0
    for sid in assignment:
        t=e['targets'][sid];train=p['targets'][sid]
        for arm in ('M','W'):
            path=a.output/'training'/arm/sid/'memory.png';sha=s.file_sha(path);image=s.read_png(path)
            for panel in ('recovery_training','qualification','application_training','mcq','application_reserved'):
                for item in t[panel]:
                    processor.begin_capture();g=s.generate(reader,processor,image,item['query'],a.device);proof=processor.end_capture()
                    s.append(out/'reads.jsonl',dict(target=sid,condition=arm,panel=panel,query=item,png_sha=sha,generation=g,score=s.score_generation(g,item,panel=='mcq'),reader_query=item['query'],processor_input=proof[0]));n+=1
                    if panel=='recovery_training' or (panel=='qualification' and item['kind']=='recovery'):
                        measured=measure_ce(a,reader,processor,image,item,termination)
                        s.append(out/'recovery-ce.jsonl',dict(target=sid,condition=arm,panel=panel,query=item,png_sha=sha,**measured));cen+=1
            for scope,value in sorted(train['state'].items()):
                for bank in ('question','instruction'):
                    for ix in range(4):
                        processor.begin_capture();item=q.recovery_form(scope,value,bank,ix);g=s.generate(reader,processor,image,item['query'],a.device);proof=processor.end_capture()
                        s.append(out/'reads.jsonl',dict(target=sid,condition=arm,panel='recovery_coverage',query=item,png_sha=sha,generation=g,score=s.score_generation(g,item),reader_query=item['query'],processor_input=proof[0]));n+=1
                        measured=measure_ce(a,reader,processor,image,item,termination)
                        s.append(out/'recovery-ce.jsonl',dict(target=sid,condition=arm,panel='recovery_coverage',query=item,png_sha=sha,**measured));cen+=1
            for item0 in t['application_training']:
                processor.begin_capture();case=old_cases[(item0['case_id'],item0['scope'],item0['value_id'])];item,_,gold=q.xml_candidates(case,0);g=s.generate(reader,processor,image,item['query'],a.device);proof=processor.end_capture()
                s.append(out/'reads.jsonl',dict(target=sid,condition=arm,panel='application_xml',query=item,png_sha=sha,generation=g,score=s.score_generation(g,{**item,'target_index':gold},True),reader_query=item['query'],processor_input=proof[0]));n+=1
            for panel in ('application_training','application_reserved'):
                for item0 in t[panel]:
                    case=old_cases[(item0['case_id'],item0['scope'],item0['value_id'])]
                    for rot in (range(4) if panel=='application_training' and item0['value_id'] in reg['contrast_values'] else (0,)):
                        item,ch,gold=s.ranking_candidates(case,rot);score=s.ranking_read(reader,processor,image,item['query'],ch,a.device);cf+=4
                        margin=score['scores'][gold]-max(x for k,x in enumerate(score['scores']) if k!=gold)
                        s.append(out/'ranking.jsonl',dict(target=sid,condition=arm,panel='application_training_rotations' if rot else panel,query=item,png_sha=sha,choices=ch,gold_index=gold,score=score,gold_margin=margin,unique_correct=margin>0));rn+=1
            for vid in sorted({item['value_id'] for item in t['application_training']}):
                for case in new_cases[vid]:
                    processor.begin_capture();item,_,gold=q.xml_candidates(case,0);g=s.generate(reader,processor,image,item['query'],a.device);proof=processor.end_capture()
                    s.append(out/'reads.jsonl',dict(target=sid,condition=arm,panel='attribute_xml',query=item,png_sha=sha,generation=g,score=s.score_generation(g,{**item,'target_index':gold},True),reader_query=item['query'],processor_input=proof[0]));n+=1
                    item,ch,gold=s.ranking_candidates(case,0);score=s.ranking_read(reader,processor,image,item['query'],ch,a.device);cf+=4
                    margin=score['scores'][gold]-max(x for k,x in enumerate(score['scores']) if k!=gold)
                    s.append(out/'ranking.jsonl',dict(target=sid,condition=arm,panel='attribute_full_action',query=item,png_sha=sha,choices=ch,gold_index=gold,score=score,gold_margin=margin,unique_correct=margin>0));rn+=1
    s.frozen(versions);s.write_frozen(out/'complete.json',dict(generations=n,rankings=rn,ranking_candidate_forwards=cf,recovery_ce_forwards=cen,processed_input_tokens=processor.total_input_tokens,
        seconds=time.monotonic()-started,artifacts={x:s.file_sha(out/x) for x in ('reads.jsonl','ranking.jsonl','recovery-ce.jsonl','identity.json')}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['register','train','evaluate'],required=True)
    for n in ('source','output','reader','base','runtime-receipt'):ap.add_argument('--'+n,type=Path)
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=4);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    if a.mode=='register':register(a)
    else:a.output.mkdir(parents=True,exist_ok=True);{'train':train,'evaluate':evaluate}[a.mode](a)
