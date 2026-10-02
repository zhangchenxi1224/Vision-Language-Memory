"""Verify the full frozen raw-condition control, without requiring new optimizer logs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

PROBE = '1f86d56fd51cfd6b96ba4cba39bcbfc26093d251'
PARENT = 'bb34092ab0d1292c87d16d9632716b218f54054b'
CHECKPOINT = '737735c4d7d3483b38be2f88d8c48fbe3050b40c8f90c0d49b21e30336455616'
RUNTIME = 'e78e86707c6e6c027ec88148c7428094c452543bd23cd174870de89c6d0b4418'
PHASE = 'training_raw_guidance1'


def collect(run, parent, bank_path, *, text_only=False):
    from scripts.reporting.collect_transition_endpoint import read, jsonl, sha
    from scripts.reporting.collect_broader_endpoint import parent_binding, phase_summary, BANK_SHA
    from vision_memory.training.latent_bank_unet import stable_seed
    from scripts.experiments.broader_writer_protocol import SEED
    bank, _, result = parent_binding(parent, bank_path, logical_sampling_commit=PARENT)
    complete, identity = read(run/'complete.json'), read(run/'identity.json')
    if complete['identity'] != identity or complete['phase'] != PHASE:
        raise ValueError('Frozen probe identity or phase differs')
    expected = {'probe_commit': PROBE, 'checkpoint_sha256': CHECKPOINT,
        'parent_runtime_sha256': RUNTIME, 'parent_result_sha256': sha(parent/'train/result.json'),
        'bank_sha256': BANK_SHA, 'optimizer_updates': 0, 'native_steps': 28,
        'guidance_scale': 1., 'image_guidance_scale': 1., 'seed': SEED, 'eval_seeds': 2,
        'groups': 151, 'raw_rows': 3020, 'matched_rows': 1510,
        'condition_style': 'cached upstream raw-event training embedding and mask, repeated into the three native branches'}
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError('Registered raw inference control changed')
    if sha(parent/'train/runtime.json') != RUNTIME or result['checkpoint_sha256'] != CHECKPOINT:
        raise ValueError('Fixed parent runtime/checkpoint differs')
    if sha(run/PHASE/'complete.json') != complete['phase_complete_sha256']:
        raise ValueError('Raw phase completion seal differs')
    names = {hashlib.sha256(group['question_id'].encode()).hexdigest()[:16]+f'-seed-{i:02d}.pt':
        (group['question_id'], stable_seed(SEED, 'heldout-evaluation-noise', i))
        for group in bank['groups'] for i in range(2)}
    if len(names) != 302:
        raise ValueError('Unexpected artifact-name collision')
    omitted, tensors = [], 0
    def verify(path, digest):
        if text_only and path.suffix == '.pt' and not path.exists():
            omitted.append(str(path.relative_to(parent if path.is_relative_to(parent) else run)))
        elif sha(path) != digest:
            raise ValueError('Changed sealed artifact: '+str(path))
    verify(parent/'train/checkpoint-final.pt', CHECKPOINT)
    summaries, pairs = {}, {}
    for label, directory, phase in (('native', parent/'train/trained', 'trained'), ('raw', run/PHASE, PHASE)):
        sealed = read(directory/'complete.json')
        if sealed['generation_rows'] != 3020 or set(sealed['artifact_hashes']) != set(names)|{'generations.jsonl', 'summary.json'}:
            raise ValueError('Incomplete full development artifact matrix')
        for name, digest in sealed['artifact_hashes'].items():
            verify(directory/name, digest)
        rows = jsonl(directory/'generations.jsonl')
        summaries[label], pairs[label] = phase_summary(rows, bank, phase)
        if not text_only:
            import torch
            from vision_memory.repro import canonical_tensor_sha256
            images = {(row['question_id'], row['noise_seed']): row['image_sha256']
                for row in rows if row['condition'] == 'matched'}
            for name, key in names.items():
                payload = torch.load(directory/name, map_location='cpu', weights_only=True)
                if (payload['question_id'], payload['noise_seed']) != key:
                    raise ValueError('Saved tensor belongs to a different condition/noise')
                trajectory, latent, pixels = payload['trajectory'], payload['latent'], payload['image']
                if (len(trajectory) != 29 or latent.shape != (1, 4, 128, 128)
                        or pixels.shape != (1, 3, 1024, 1024)
                        or any(value.dtype != torch.float32 or not torch.isfinite(value).all()
                            for value in [latent, pixels, *trajectory])
                        or any(value.shape != latent.shape for value in trajectory)):
                    raise ValueError('Expected all 29 finite FP32 native states and decoded pixels')
                noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(key[1]), dtype=torch.float32)
                if (not torch.equal(trajectory[0], noise) or not torch.equal(trajectory[-1], latent)
                        or canonical_tensor_sha256(pixels) != images[key]
                        or pixels.min() < 0 or pixels.max() > 1):
                    raise ValueError('Actual Gaussian start, endpoint or Reader pixels differ')
                tensors += 1
    changes = Counter()
    for key, before in pairs['native'].items():
        after = pairs['raw'][key]
        if key[1] == 'matched':
            changes[f"{int(before['passed'])}->{int(after['passed'])}"] += 1
        elif before != after:
            raise ValueError('Fixed negative-control tokens/pixels changed')
    passed = summaries['raw']['correct_eos'] == 1510
    if (complete['native_summary'] != summaries['native'] or complete['raw_summary'] != summaries['raw']
            or complete['matched_changes'] != changes or complete['development_all_correct_eos'] != passed):
        raise ValueError('Complete result disagrees with every raw token/EOS pair')
    if not text_only:
        from scripts.train.train_latent_bank_unet import source_hashes
        if source_hashes() != identity['source_hashes']:
            raise ValueError('Probe training/runtime source differs from pinned checkout')
    return {'identity': identity, 'complete_sha256': sha(run/'complete.json'),
        'native_summary': summaries['native'], 'raw_summary': summaries['raw'],
        'matched_changes': changes, 'development_all_correct_eos': passed,
        'raw_rows_recounted': 6040, 'current_and_parent_pts_checked': tensors,
        'artifacts_omitted_locally': omitted,
        'scope': 'All fixed observed development cells. Actual checkpoint, 604 PTs, pure Gaussian starts, finite 29-state paths and Reader pixels verified on CPU. Portable archive carries full raw records and seals; large PTs remain remote. Independent expressions, RGB chains and CLI remain required.'}


def portable_files(run, parent, bank):
    files = {'bank/manifest.json': bank}
    for name in ('identity.json', 'complete.json', PHASE+'/complete.json', PHASE+'/generations.jsonl', PHASE+'/summary.json'):
        files['run/'+name] = run/name
    for name in ('preregistered-experiment.json', 'terminal.json', 'train/identity.json', 'train/runtime.json',
            'train/result.json', 'train/baseline-reference-check.json', 'train/trained/complete.json',
            'train/trained/generations.jsonl', 'train/trained/summary.json'):
        files['parent/'+name] = parent/name
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    sys.path[:0] = [str(args.source_root), str(args.source_root/'src')]
    from scripts.reporting.collect_transition_endpoint import read, sha
    if args.archive:
        from scripts.reporting.verify_broader_outputs_local import unpack, compare_recount
        with tempfile.TemporaryDirectory(prefix='raw-condition-recount-') as temporary:
            root = Path(temporary)
            unpack(args.archive, args.sha256, root)
            remote = read(root/'verified-summary.json')
            for name, digest in remote['portable_sha256'].items():
                path = (root/name).resolve()
                if not path.is_relative_to(root.resolve()) or sha(path) != digest:
                    raise ValueError('Portable parent/probe file seal differs')
            summary = collect(root/'run', root/'parent', root/'bank/manifest.json', text_only=True)
            compare_recount(remote, summary, {'portable_sha256', 'current_and_parent_pts_checked', 'artifacts_omitted_locally'})
            if remote['current_and_parent_pts_checked'] != 604:
                raise ValueError('Missing complete independent remote tensor verification')
            summary = {'archive_sha256': args.sha256, 'raw_rows_recounted': 6040,
                'native_correct_eos': summary['native_summary']['correct_eos'],
                'raw_correct_eos': summary['raw_summary']['correct_eos'], 'matched_changes': summary['matched_changes'],
                'development_all_correct_eos': summary['development_all_correct_eos'],
                'scope': 'All 6040 portable raw rows and seals recounted locally.604 PTs and checkpoint checked remotely; this local archive does not contain them. Development is not full functional validation.'}
    else:
        if not all((args.run, args.parent, args.bank)):
            parser.error('Remote collection requires run, parent and bank')
        if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.source_root, text=True).strip() != PROBE
                or subprocess.check_output(['git', 'status', '--porcelain'], cwd=args.source_root, text=True).strip()):
            raise ValueError('Require clean fixed raw-control checkout')
        import torch
        torch.set_num_threads(1)
        summary = collect(args.run, args.parent, args.bank)
        files = portable_files(args.run, args.parent, args.bank)
        summary['portable_sha256'] = {name: sha(path) for name, path in files.items()}
    output = Path(str(args.output_prefix)+'-summary.json')
    archive_path = Path(str(args.output_prefix)+'-evidence.tgz')
    if output.exists() or (not args.archive and archive_path.exists()):
        raise ValueError('Collection output already exists')
    output.write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    if not args.archive:
        with tarfile.open(archive_path, 'w:gz') as archive:
            for name, path in files.items():
                archive.add(path, arcname=name)
            archive.add(output, arcname='verified-summary.json')
        print(json.dumps({'archive_sha256': sha(archive_path), 'summary_sha256': sha(output),
            'raw_correct_eos': summary['raw_summary']['correct_eos'], 'matched_changes': summary['matched_changes']}))
    else:
        print(json.dumps(summary))


if __name__ == '__main__':
    main()
