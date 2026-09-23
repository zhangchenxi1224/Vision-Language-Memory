"""Upload immutable run weights from the shared disk; authentication is stdin-only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request_json(url, headers, stream=None):
    request=Request(url,headers=headers,data=stream,method='POST' if stream is not None else 'GET')
    try:
        with urlopen(request,timeout=3600 if stream is not None else 30) as response:
            return response.status,json.load(response)
    except HTTPError as error:
        raise RuntimeError(f'GitHub HTTP {error.code}') from None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--release-id',type=int,required=True)
    parser.add_argument('--kind',choices=('writers','student-pngs'),default='writers')
    args=parser.parse_args()
    token=sys.stdin.readline().strip()
    if not token:raise RuntimeError('Missing transient GitHub authentication')
    headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
             'User-Agent':'Vision-Language-Memory-artifact-upload'}
    repository='zhangchenxi1224/Vision-Language-Memory'
    release=f'https://api.github.com/repos/{repository}/releases/{args.release_id}'
    status,response=request_json(release,headers)
    if status!=200:raise RuntimeError(f'Release lookup HTTP {status}')
    assets={x['name']:x for x in response['assets']}
    receipt='writer-release-upload' if args.kind=='writers' else 'student-png-release-upload'

    def upload(arm):
        if args.kind=='writers':
            folder=args.run/'writers'/arm/'write'
            path=folder/'checkpoint-final.pt'
            expected=json.loads((folder/'complete.json').read_text())['checkpoint_sha256']
            name=f'writer-{arm}-write-2048.pt'
        else:
            folder=args.run/'rollouts'/arm/'write'
            markers=sorted(folder.glob('*/seed-*/complete.json'))
            if len(markers)!=308:raise RuntimeError('Require all 154 x2 write PNG endpoints')
            name=f'student-rgb-{arm}-write.tar'
            path=args.run/'archives'/name
            path.parent.mkdir(exist_ok=True)
            if not path.exists():
                temporary=path.with_suffix('.tmp.tar')
                with tarfile.open(temporary,'w') as archive:
                    for marker in markers:
                        done=json.loads(marker.read_text())
                        if len(done['images'])!=1:raise RuntimeError('Expected single-write endpoints')
                        png=marker.parent/'memory-00.png'
                        if hashlib.sha256(png.read_bytes()).hexdigest()!=done['images'][0]['png_sha256']:
                            raise RuntimeError('PNG differs from rollout completion marker')
                        for item in (png,marker):archive.add(item,arcname=str(item.relative_to(args.run)))
                temporary.replace(path)
            expected=None
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
        if expected is not None and digest.hexdigest()!=expected:raise RuntimeError(f'Checkpoint hash changed for {arm}')
        expected=digest.hexdigest()
        if name in assets:
            asset=assets[name]
        else:
            print(json.dumps(dict(upload_started=name,bytes=path.stat().st_size)),flush=True)
            with path.open('rb') as stream:
                status,asset=request_json(f'https://uploads.github.com/repos/{repository}/releases/{args.release_id}/assets?name={name}',
                    {**headers,'Content-Type':'application/octet-stream',
                    'Content-Length':str(path.stat().st_size)},stream)
            if status!=201:raise RuntimeError(f'Asset upload HTTP {status} for {arm}')
        if asset['state']!='uploaded' or asset['size']!=path.stat().st_size:
            raise RuntimeError(f'Inspect incomplete release asset: {name}')
        if asset.get('digest')!='sha256:'+expected:
            raise RuntimeError(f'GitHub asset digest missing or mismatched: {name}')
        result=dict(arm=arm,name=name,asset_id=asset['id'],bytes=asset['size'],
            sha256=expected,github_digest=asset['digest'],url=asset['browser_download_url'],finished_unix=time.time())
        (args.run/f'{receipt}-{arm}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(upload,('A','B')))
    (args.run/f'{receipt}.json').write_text(json.dumps(dict(assets=results),indent=2)+'\n')


if __name__=='__main__':main()
