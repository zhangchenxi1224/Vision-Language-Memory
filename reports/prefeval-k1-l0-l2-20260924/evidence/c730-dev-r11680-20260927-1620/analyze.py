from pathlib import Path
import json,sys,hashlib,collections,datetime,os,shutil,tarfile
p=Path('/inspire/ssd/project/exploration-topic/czxs26210936');r=p/'runs/prefeval-b-mcq-20260925';repo=p/'repos/prefeval-b-refresh-5559681';out=r/'c730-dev-r11680-20260927-1620'
sys.path.insert(0,str(repo))
from scripts.experiments.prefeval_k1_data import load_records,official_mcq,sha,option_order
parse=official_mcq(repo/'third_party/prefeval_reference')['extract_choice']
ids=[x['base_pair_id'] for x in load_records('dev')];F=['T1','T2','T3','O1','O2'];D=[0,1,5,10]
a=r/'retain730-C';rows=[json.loads(x) for x in (a/'readback-final-dev-V1/readback-0.jsonl').read_text().splitlines()]
idx={(x['pair_id'],x['chain'],x['prefix'],x['control'],x['family']):x for x in rows}
expected={(i,c,d,k,f) for i in ids for c in range(2) for d in D for k in ['memory','mismatch'] for f in F}
expected|={(i,0,d,'text',f) for i in ids for d in D for f in F}
expected|={(i,0,0,'blank',f) for i in ids for f in F}
assert len(idx)==len(rows)==9450 and set(idx)==expected
checks={'rows':9450,'position_rows':0,'pngs':0,'chains':0,'parse_failures':0,'truncations':0}
cp=sha(a/'train/checkpoint-final.pt')
assert cp=='3d67d91f4e26bebc503f95e697d61b9d13453738be5d8a4e90e898cef4372ced'
for done in (a/'final-dev-V1').glob('*/seed-*/complete.json'):
 info=json.loads(done.read_text());writes=[json.loads(x) for x in (done.parent/'writes.jsonl').read_text().splitlines()]
 assert info['binding']['checkpoint_sha256']==cp and len(writes)==11
 for i,w in enumerate(writes):
  png=done.parent/('prefix-%02d.png'%i)
  assert sha(png)==info['png_hashes'][png.name]==w['output_png_sha256']
  if i:assert w['source_png_sha256']==writes[i-1]['output_png_sha256']
  checks['pngs']+=1
 checks['chains']+=1
assert checks['pngs']==1980 and checks['chains']==180
for x in rows:
 pred=parse(x['generated']['raw']);assert pred==x['predicted_letter']
 assert x['correct_letter']=='ABCD'[x['option_order'].index(0)]
 assert x['correct']==(pred==x['correct_letter'])
 checks['parse_failures']+=pred is None;checks['truncations']+=x['generated']['truncated']
 if x['control'] in ['memory','mismatch']:
  source=x['pair_id'] if x['control']=='memory' else x['donor_pair_id']
  assert source in ids and (source!=x['pair_id'] if x['control']=='mismatch' else True)
  path=a/'final-dev-V1'/source.replace(':','_')/('seed-%d'%x['chain'])/('prefix-%02d.png'%x['prefix'])
  assert str(path)==x['png_path'] and sha(path)==x['png_sha256']
 def step(pid,f):return int.from_bytes(hashlib.sha256(('eval:'+pid+':'+f).encode()).digest()[:4],'big')
 assert x['option_order']==option_order(x['pair_id'],step(x['pair_id'],x['family']))[0]
def correct(i,c,d,k,f):return idx[i,c,d,k,f]['correct']
stats=[]
for d in D:
 for f in F:
  pairs=[(correct(i,c,d,'memory',f),correct(i,c,d,'mismatch',f)) for i in ids for c in range(2)]
  stats.append({'depth':d,'family':f,'matched':sum(m for m,n in pairs),'mismatch':sum(n for m,n in pairs),'net':sum(m-n for m,n in pairs),'gray':sum(correct(i,0,0,'blank',f) for i in ids),'text':sum(correct(i,0,d,'text',f) for i in ids),'repair':sum(m and not n for m,n in pairs),'regress':sum(n and not m for m,n in pairs),'both_noise':sum(all(correct(i,c,d,'memory',f) for c in range(2)) for i in ids)})
joint={'ood_by_depth':{d:{'same_image':sum(all(correct(i,c,d,'memory',f) for f in ['O1','O2']) for i in ids for c in range(2)),'both_noise':sum(all(correct(i,c,d,'memory',f) for f in ['O1','O2'] for c in range(2)) for i in ids)} for d in D},'registered_depths_and_both_noise':{f:sum(all(correct(i,c,d,'memory',f) for c in range(2) for d in D) for i in ids) for f in F},'ood_registered_depths_and_both_noise':sum(all(correct(i,c,d,'memory',f) for c in range(2) for d in D for f in ['O1','O2']) for i in ids)}
pos={}
for d in [0,10]:
 pr=[json.loads(x) for x in (a/('positions-final-dev-V1-%d.jsonl'%d)).read_text().splitlines()]
 pi={(x['pair_id'],x['chain'],x['position']):x for x in pr}
 assert len(pi)==len(pr)==900 and set(pi)=={(i,c,q) for i in ids for c in range(2) for q in ['official',0,1,2,3]}
 for x in pr:
  pred=parse(x['generated']['raw'])
  assert pred==x['predicted_letter'] and x['correct_letter']=='ABCD'[x['order'].index(0)] and x['correct']==(pred==x['correct_letter'])
  assert x['png_sha256']==idx[x['pair_id'],x['chain'],d,'memory','T1']['png_sha256']
  checks['parse_failures']+=pred is None;checks['truncations']+=x['generated']['truncated']
 checks['position_rows']+=len(pr)
 pos[d]={'per_position':{str(q):sum(pi[i,c,q]['correct'] for i in ids for c in range(2)) for q in ['official',0,1,2,3]},'all_four':sum(all(pi[i,c,q]['correct'] for q in range(4)) for i in ids for c in range(2)),'all_four_both_noise':sum(all(pi[i,c,q]['correct'] for q in range(4) for c in range(2)) for i in ids)}
summary={'checks':checks,'metrics':stats,'joint':joint,'positions':pos,'checkpoint_sha256':cp}
(out/'C-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
train=r/'retain730-R/train'
op=[json.loads(x) for x in (train/'optimization.jsonl').read_text().splitlines()]
assert [x['step'] for x in op]==list(range(1,11681))
exposure=collections.Counter();depths=collections.defaultdict(set)
for x in op:
 assert len(x['draws'])==4 and sum(t['position']==0 for t in x['draws'])==2
 rn=(x['step']-1)//5840
 for t in x['draws']:
  kind='write' if t['position']==0 else 'retain'
  exposure[(t['pair_id'],rn,kind)]+=1
  if kind=='retain':
   assert t['source_round']==rn and t['source_index']==t['position']-1
   depths[t['pair_id'],rn].add(t['source_index'])
assert len(exposure)==730*2*2 and set(exposure.values())=={16}
assert all(v==set(range(10)) for v in depths.values())
import torch
resume=torch.load(train/'resume.pt',map_location='cpu',mmap=True,weights_only=False)
assert resume['optimizer_step']==11680 and resume['episode_cursor']==46720
osteps={int(v['step'].item()) for v in resume['optimizer']['state'].values()};assert osteps=={11680}
weight=sha(train/'checkpoint-step-011680.pt');done=json.loads((train/'round-1-complete.json').read_text());assert done['checkpoint_sha256']==weight
rs={'steps':11680,'draws':46720,'per_pref_per_round_write_retain':16,'depths_all_0_9':True,'checkpoint_sha256':weight,'resume_sha256':sha(train/'resume.pt'),'resume_step':resume['optimizer_step'],'episode_cursor':resume['episode_cursor'],'optimizer_parameters':len(resume['optimizer']['state']),'optimizer_steps':list(osteps),'rng_keys':list(resume['rng_state']),'trainer_state':resume['trainer_state'],'round2':[]}
for shard in (r/'retain730-R/sources/round-2').glob('shard-*'):
 m=json.loads((shard/'manifest.json').read_text())
 assert m['checkpoint_sha256']==weight and m['initial_variant']==0
 rs['round2'].append({'shard':shard.name,'chains':len(list(shard.glob('*/seed-*/complete.json'))),'pngs':len(list(shard.rglob('*.png'))),'manifest':m})
(out/'R-summary.json').write_text(json.dumps(rs,indent=2)+'\n')
print(json.dumps(summary));print('R',json.dumps({k:v for k,v in rs.items() if k!='round2'}));print('SHARDS',[(x['shard'],x['chains'],x['pngs']) for x in rs['round2']])
