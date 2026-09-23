"""Archive both complete fixed-budget banks, including failures and raw training logs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    root=a.run;out=root/'archives';out.mkdir(exist_ok=True)
    manifest=dict(kind='all128_fixed_teacher_endpoints',banks={},raw_training=[],limits=[
        'These are per-state optimized teachers, not shared Writer performance.',
        'Original-generation.json is an unscored diagnostic; official-format evaluation is separate.'])
    raw=[]
    for arm in ('A','B'):
        folder=root/'teachers'/arm
        if not all((folder/f'complete-{i}.json').exists() for i in (0,1)):raise RuntimeError('Bank still running')
        endpoints=sorted(folder.glob('*/complete.json'))
        if len(endpoints)!=64:raise RuntimeError('Keep all64 endpoints; no success filtering')
        samples=[]
        for endpoint in endpoints:
            samples.append(dict(id=endpoint.parent.name,endpoint=json.loads(endpoint.read_text())))
        archive=out/f'teacher-bank-{arm}.tgz'
        if not archive.exists():
            temp=archive.with_suffix('.tmp')
            with tarfile.open(temp,'w:gz') as tar:tar.add(folder,arcname=f'teachers/{arm}')
            temp.replace(archive)
        manifest['banks'][arm]=dict(archive=archive.name,bytes=archive.stat().st_size,sha256=digest(archive),states=samples)
        for file in sorted(folder.rglob('*')):
            if file.is_file() and file.suffix in ('.json','.jsonl'):
                raw.append(dict(path=str(file.relative_to(root)),text=file.read_text(encoding='utf-8')))
    for file in sorted(root.glob('*.json')):
        if file.name.startswith(('dispatch-','failure-')):
            raw.append(dict(path=file.name,text=file.read_text(encoding='utf-8')))
    for file in sorted((root/'pipeline').glob('*/running.json')):
        raw.append(dict(path=str(file.relative_to(root)),text=file.read_text(encoding='utf-8')))
    rawpath=out/'teacher-raw-records.jsonl.gz'
    with gzip.open(rawpath,'wt',encoding='utf-8') as stream:
        for row in raw:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest['raw_training']=dict(archive=rawpath.name,files=len(raw),sha256=digest(rawpath))
    (out/'teacher-bank-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({arm:{k:v for k,v in bank.items() if k!='states'} for arm,bank in manifest['banks'].items()}))

if __name__=='__main__':main()
