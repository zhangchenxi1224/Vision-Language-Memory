"""Upload immutable run weights from the shared disk; authentication is stdin-only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import requests


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--release-id',type=int,required=True)
    args=parser.parse_args()
    token=sys.stdin.readline().strip()
    if not token:raise RuntimeError('Missing transient GitHub authentication')
    headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'}
    repository='zhangchenxi1224/Vision-Language-Memory'
    release=f'https://api.github.com/repos/{repository}/releases/{args.release_id}'
    response=requests.get(release,headers=headers,timeout=30)
    if response.status_code!=200:raise RuntimeError(f'Release lookup HTTP {response.status_code}')
    assets={x['name']:x for x in response.json()['assets']}

    def upload(arm):
        folder=args.run/'writers'/arm/'write'
        path=folder/'checkpoint-final.pt'
        expected=json.loads((folder/'complete.json').read_text())['checkpoint_sha256']
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
        if digest.hexdigest()!=expected:raise RuntimeError(f'Checkpoint hash changed for {arm}')
        name=f'writer-{arm}-write-2048.pt'
        if name in assets:
            asset=assets[name]
        else:
            print(json.dumps(dict(upload_started=name,bytes=path.stat().st_size)),flush=True)
            with path.open('rb') as stream:
                reply=requests.post(f'https://uploads.github.com/repos/{repository}/releases/{args.release_id}/assets',
                    params={'name':name},headers={**headers,'Content-Type':'application/octet-stream',
                    'Content-Length':str(path.stat().st_size)},data=stream,timeout=(30,3600))
            if reply.status_code!=201:raise RuntimeError(f'Asset upload HTTP {reply.status_code} for {arm}')
            asset=reply.json()
        if asset['state']!='uploaded' or asset['size']!=path.stat().st_size:
            raise RuntimeError(f'Inspect incomplete release asset: {name}')
        if asset.get('digest')!='sha256:'+expected:
            raise RuntimeError(f'GitHub asset digest missing or mismatched: {name}')
        result=dict(arm=arm,name=name,asset_id=asset['id'],bytes=asset['size'],
            sha256=expected,github_digest=asset['digest'],url=asset['browser_download_url'],finished_unix=time.time())
        (args.run/f'writer-release-upload-{arm}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(upload,('A','B')))
    (args.run/'writer-release-upload.json').write_text(json.dumps(dict(assets=results),indent=2)+'\n')


if __name__=='__main__':main()
