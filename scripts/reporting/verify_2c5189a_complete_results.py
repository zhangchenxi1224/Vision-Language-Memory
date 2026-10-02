import hashlib,json,sys,tarfile,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.reporting.verify_broader_outputs_local import unpack,compare_recount
from scripts.reporting.collect_transition_endpoint import read,sha
from scripts.reporting.collect_broader_validation import collect
from scripts.reporting.verify_png_readback_local import verify as png_verify
from scripts.probes.rgb_package_parity import verify as cli_verify
from PIL import Image
OUT=ROOT/'reports/official-alignment-results-20260913'
PARTS=OUT/'2c5189a-complete-results-sync.parts'
m=read(PARTS/'manifest.json')
assert sha(PARTS/'manifest.json')=='91778ac0bc82660246c4b73d365a3dbad26e6fe6413a4f8cc0cae22447b3c6b5'
bundle=ROOT/'.cache/2c5189a-results.tar'
if not bundle.exists():
 with bundle.open('wb') as output:
  for chunk in m['chunks']:
   part=PARTS/chunk['file']
   assert part.stat().st_size==chunk['bytes'] and sha(part)==chunk['sha256']
   output.write(part.read_bytes())
assert sha(bundle)==m['sha256']
V='2c5189a0847acd6653b687031ab13e6cd4cfc53f';T='b62ec027ad725aeb6ecc772aa85e7a3ff6e49b36'
with tempfile.TemporaryDirectory(dir=ROOT/'.cache',prefix='latest-sync-') as tmp:
 root=Path(tmp)
 with tarfile.open(bundle) as tar:
  seen=set()
  for item in tar:
   p=(root/item.name).resolve()
   if not item.isfile() or not p.is_relative_to(root.resolve()) or p in seen:raise ValueError('Unsafe/duplicate bundle entry')
   seen.add(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(item).read())
 spec=read(root/'files.json')
 for item in spec['files']:
  p=root/item['file']
  assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 (OUT/'2c5189a-complete-results-file-manifest.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
 for p in root.glob('*.json'):
  if p.name!='files.json':(OUT/p.name).write_bytes(p.read_bytes())
 parent=root/'parent'
 unpack(OUT/'2c5189a-logical-endpoint-evidence.tgz','d59aa426e5a6e093d5887bd305b7d2204276a10956b27f3aab3e78d267348f67',parent)
 results={'bundle_sha256':m['sha256'],'files_verified':len(spec['files']),'validation':{},'png':{},'cli':{}}
 for suite,prefix in [('registered','2c5189a-logical'),('fresh_wording_v1','2c5189a-fresh-wording')]:
  for lane in ['confirmation','chains','prefix0','prefix1']:
   name=prefix+'-'+lane
   archive=root/(name+'-evidence.tgz');run=root/(name+'-extracted')
   unpack(archive,sha(archive),run)
   actual=collect(run,parent,parent/'bank/manifest.json',V,text_only=True,logical_sampling_commit=T,validation_set=suite)
   compare_recount(read(run/'verified-summary.json'),actual,{'artifacts_omitted_locally','all_files_verified_here','pixel_noise_trajectory_tensors_checked'})
   images=list(run.glob('*.png'))
   assert len(images)==actual['generated_images']
   for p in images:
    with Image.open(p) as im:
     assert im.mode=='RGB' and im.size==(1024,1024);im.verify()
   results['validation'][name]={k:actual[k] for k in ['matched_correct_eos','matched_rows','raw_rows','generated_images','all_generated_correct_eos'] if k in actual}
   png_name='2c5189a-png-readback-'+suite+'-'+lane
   pa=root/(png_name+'-evidence.tgz')
   result=png_verify(pa,sha(pa),archive,sha(archive),V,continuation_validation_commit=V,continuation_training_commit=T)
   (OUT/(png_name+'-local-verification.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
   results['png'][png_name]={k:result['recount'][k] for k in ['png','png_images_verified_here','paired_matched','chain_parity_passed']}
   print(json.dumps({'verified':name,'scores':results['validation'][name]}),flush=True)
  prepared=root/(prefix+'-parity');inference=root/(prefix+'-inference')
  recorded=read(prepared/'parity-result.json');actual=cli_verify(prepared,inference)
  assert actual==recorded
  assert sha(root/(prefix+'-package')/'manifest.json')==actual['package_manifest_sha256']
  results['cli'][prefix]=actual
 results['scope']='All eight complete raw functional matrices and PNGs, all eight PNG readback matrices and both six-write/thirty-read CLI replays verified locally. Model weights and large PT tensor payloads remain remote. No semantic rescoring.'
 (OUT/'2c5189a-complete-results-local-verification.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
 print('ALL_VERIFIED',flush=True)