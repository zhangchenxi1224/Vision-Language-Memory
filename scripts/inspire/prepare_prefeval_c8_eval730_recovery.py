"""Copy only validated complete single-write chains into an independent recovery directory."""
import hashlib
import importlib.util
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
TASK = PROJECT/'runs/prefeval-multitarget-20260927'
OLD = TASK/'eval730/fixed-c8-b0-20260928'
NEW = TASK/'eval730/recovery-20260928-1920'
CONTROLLER = PROJECT/'repos/prefeval-c8-eval730-20260928'
PROTOCOL_SHA = 'c670e3db2d7d54bf449999cf693d7544fb54bbc7040420b641c31484e6f641dc'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert not NEW.exists(), 'Recovery destination must be new'
    assert not (OLD/'complete.json').exists()
    assert not list(OLD.rglob('readback-*.jsonl')), 'This recovery is generation-only'
    assert sha(CONTROLLER/'scripts/inspire/run_prefeval_c8_eval730.py') == 'ab4d702b41115bcfd888d5ef7cb561577eaf8af88b035830395ebbe45be2a8cf'
    assert sha(OLD/'protocol.json') == PROTOCOL_SHA
    spec = importlib.util.spec_from_file_location('frozen_eval730', CONTROLLER/'scripts/inspire/run_prefeval_c8_eval730.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = module.prepare(NEW)  # Rehash both checkpoints, worker checkout and original data.
    assert sha(NEW/'protocol.json') == PROTOCOL_SHA
    assert sha(NEW/'ids.json') == sha(OLD/'ids.json')
    frozen = PROJECT/'repos/prefeval-multitarget-round2'
    sys.path[:0] = [str(frozen), str(frozen/'src')]
    from scripts.experiments.prefeval_k1_data import load_records, event_text
    from scripts.experiments.prefeval_k1_variants import load_variants, apply_variant
    from scripts.experiments.prefeval_k1_source_bank import noise_namespace
    from vision_memory.training.latent_bank_unet import stable_seed
    rows = load_records('train')
    variants = load_variants(module.VARIANTS, rows)
    rowmap = {(r['base_pair_id'],v):apply_variant(r,variants,v) for r in rows for v in range(2)}
    foldermap = {x.replace(':','_'):x for x in protocol['ids']}
    receipt = {'created_utc':datetime.now(timezone.utc).isoformat(),'source':str(OLD),'destination':str(NEW),
               'protocol_sha256':PROTOCOL_SHA,'checkpoints':protocol['checkpoints'],
               'copied_sha256':{},'complete_chains':{},'partial_chains_not_copied':[],
               'audit_script_sha256':sha(Path(__file__)),
               'policy':'Preserve complete PNG/writes/complete byte-for-byte. Rebuild any incomplete chain independently; never append to old writes.'}
    for arm in ['C8','B0']:
        for v in range(2):
            source = OLD/arm/f'eval-V{v}'
            if not source.exists():
                continue
            target = NEW/arm/f'eval-V{v}'
            binding = {'checkpoint_sha256':protocol['checkpoints'][arm]['sha256'],'split':'train',
                       'steps':28,'cfg':1,'noise_chains':8,'inter_turns':0,
                       'state':'only reopened uint8 RGB PNG; fresh Gaussian each write',
                       'noise_domain':'mt8-eval','initial_variants_sha256':sha(module.VARIANTS),'initial_variant':v}
            target.mkdir(parents=True)
            for manifest in source.glob('manifest*.json'):
                assert json.loads(manifest.read_text()) == binding
                shutil.copyfile(manifest,target/manifest.name)
                assert sha(manifest) == sha(target/manifest.name)
            count = 0
            for chain_folder in sorted(source.glob('*/seed-*')):
                done_path = chain_folder/'complete.json'
                if not done_path.exists():
                    receipt['partial_chains_not_copied'].append(str(chain_folder))
                    continue
                pid = foldermap[chain_folder.parent.name]
                chain = int(chain_folder.name.split('-')[1])
                assert 0 <= chain < 8
                done = json.loads(done_path.read_text())
                assert done['binding'] == binding and set(done['png_hashes']) == {'prefix-00.png'}
                png = chain_folder/'prefix-00.png'
                assert sha(png) == done['png_hashes']['prefix-00.png']
                writes = [json.loads(x) for x in (chain_folder/'writes.jsonl').read_text().splitlines()]
                assert len(writes) == 1
                w = writes[0]
                assert w['position'] == 0 and w['source_png_sha256'] is None
                assert w['output_png_sha256'] == sha(png)
                assert w['event'] == event_text(rowmap[pid,v]['history'][:2])
                assert w['noise_seed'] == stable_seed(20260924,noise_namespace(pid,chain,'mt8-eval'),0)
                destination = target/chain_folder.relative_to(source)
                destination.mkdir(parents=True)
                for name in ['prefix-00.png','writes.jsonl','complete.json']:
                    original = chain_folder/name
                    copied = destination/name
                    shutil.copyfile(original,copied)
                    actual = sha(original)
                    assert sha(copied) == actual
                    receipt['copied_sha256'][str(copied.relative_to(NEW))] = actual
                count += 1
            receipt['complete_chains'][f'{arm}/V{v}'] = count
    assert receipt['complete_chains'] == {'C8/V0':765}, receipt['complete_chains']
    evidence = [p for p in OLD.rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.log']]
    evidence += [TASK/'eval730/launch.json',TASK/'eval730/pipeline.log',TASK/'eval730/runtime-usercheck-20260928-1900.json']
    archive = TASK/'eval730-interruption-20260928-1900.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(set(evidence)):
            tar.add(path,arcname=str(path.relative_to(TASK)),recursive=False)
    receipt['interruption_archive'] = {'path':str(archive),'sha256':sha(archive),'files':len(set(evidence))}
    (NEW/'recovery-preparation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'copied_sha256'},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()

