"""Export selected FP32 tensor storage, without loading models or image storage.

The CPU gateway has no torch. This restricted metadata reader supports only
the tensor rebuild globals found in our audited torch.save artifacts. It never
executes arbitrary pickle globals. Original files are read only.
"""
import argparse
import collections
import hashlib
import io
import json
from pathlib import Path
import pickle
import zipfile


def rebuild(storage, offset, shape, stride, *unused):
    return dict(tensor=True, storage=storage, offset=offset, shape=shape, stride=stride)


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
            return rebuild
        if module == 'torch' and name == 'FloatStorage':
            return 'float32'
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        raise ValueError(f'Unsupported pickle global {module}.{name}')

    def persistent_load(self, value):
        kind, dtype, key, location, count = value
        if kind != 'storage' or dtype != 'float32':
            raise ValueError('Only FP32 tensor storage is supported')
        return dict(key=key, dtype=dtype, count=count)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    p=args.root
    bank_path=p/'runs/oracle-to-unet/f5c9c9e-direct-multiprompt-dl-base-seed20260908/bank/manifest.json'
    bank=json.loads(bank_path.read_text())
    distill=p/'runs/reference-distillation/3702387-direct96-dl-base-r02'
    learn=p/'runs/unet-learnability/fd07eb1-20260909-r01'
    metadata=dict(bank_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest(),tensors={},json={},inputs={})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as output:
        def extract(path, selections):
            with zipfile.ZipFile(path) as archive:
                meta=next(n for n in archive.namelist() if n.endswith('/data.pkl'))
                prefix=meta[:-len('data.pkl')]
                data=MetadataReader(io.BytesIO(archive.read(meta))).load()
                if prefix+'byteorder' in archive.namelist():
                    assert archive.read(prefix+'byteorder')==b'little'
                metadata['inputs'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                for name,keys in selections:
                    tensor=data
                    for key in keys: tensor=tensor[key]
                    assert tensor['tensor'] and tensor['storage']['dtype']=='float32'
                    raw=archive.read(prefix+'data/'+tensor['storage']['key'])
                    assert len(raw)==tensor['storage']['count']*4
                    member='storage/'+hashlib.sha256(raw).hexdigest()+'.bin'
                    if member not in output.namelist(): output.writestr(member,raw)
                    metadata['tensors'][name]={**tensor,'archive_storage':member,'source_path':str(path)}
        for member in bank['teachers']:
            extract(Path(member['latent_path']),[(f"teacher/{member['teacher_id']}",[])])
        extract(Path(bank['groups'][0]['source_latent_path']),[('source',[])])
        for phase in ['reference','baseline','step-000064','step-000256','trained']:
            for path in sorted((distill/phase).glob('*-seed-*.pt')):
                i=int(path.stem.rsplit('-',1)[-1])
                selections=[(f'{phase}/{i}/latent',['latent'])]
                if phase in ['reference','baseline','trained']:
                    selections += [(f'{phase}/{i}/trajectory/{j}',['trajectory',j]) for j in range(5)]
                extract(path,selections)
            metadata['json'][phase+'/summary']=json.loads((distill/phase/'summary.json').read_text())
            rows=[json.loads(x) for x in (distill/phase/'generations.jsonl').read_text().splitlines()]
            metadata['json'][phase+'/answers']=[{k:r[k] for k in ['condition','noise_seed','prompt_id','raw','answer_ce','eos_ce','scorer']} for r in rows]
        metadata['json']['distill/result']=json.loads((distill/'result.json').read_text())
        metadata['json']['distill/training']=[json.loads(x) for x in (distill/'training.jsonl').read_text().splitlines()]
        metadata['json']['learn/terminal']=json.loads((learn/'terminal.json').read_text())
        for rank,step in [(4,512),(4,2048),(16,512)]:
            run=learn/f'single-rank{rank}'
            extract(run/'outputs'/f'step{step:04d}-train-000.pt',[(f'single/rank{rank}/step{step}/train',['latent'])])
            for kind in ['identity','cases','baseline-rms']:
                value=json.loads((run/(kind+'.json')).read_text())
                if kind=='identity': value.pop('source_hashes',None)
                metadata['json'][f'single/rank{rank}/{kind}']=value
            metadata['json'][f'single/rank{rank}/step{step}/result']=json.loads((run/f'result-{step:04d}.json').read_text())
        output.writestr('metadata.json',json.dumps(metadata))
    print(json.dumps(dict(output=str(args.output),bytes=args.output.stat().st_size,tensors=len(metadata['tensors']))))


if __name__=='__main__': main()
