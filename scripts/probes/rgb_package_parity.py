"""Prepare and verify a fixed six-write replay through the bank-free inference CLI.

Preparation reads sealed probe evidence. The separate inference process receives
only commands.jsonl, a parameter package, and the pinned models/source paths.
This is engineering parity, even when both implementations answer incorrectly.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
PROMPTS = ('original_open', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3', 'paraphrase_4')
FIELDS = ('raw', 'raw_with_special_tokens', 'input_token_ids', 'generated_token_ids',
          'eos_token_ids', 'eos_reached', 'truncated', 'finish_reason')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def prepare(reference, package, output, *, four_gpu_warm_start=False):
    from scripts.probes.transition_validation_plan import plan
    from vision_memory.dreamlite.writer_package import inspect_package
    reference, package, output = map(Path, (reference, package, output))
    manifest = inspect_package(package)
    complete = read_json(reference / 'complete.json')
    identity = complete['identity']
    if four_gpu_warm_start:
        from scripts.experiments.transition_warm_start_plan import plan as warm_plan
        registered = warm_plan()['validation']
        parent_commit = '046c1f1d1c398dbd08578d7c4ba6814343fea0d5'
    else:
        registered = plan(20260913)
        parent_commit = '9628d7142db5a81a9d11a35b89d0515ef32d2e4f'
    if (identity['mode'] != 'rgb_chains' or identity['registered_plan'] != registered
            or identity['parent_commit'] != parent_commit
            or manifest['parent_checkpoint_sha256'] != identity['checkpoint_sha256']
            or manifest['parent_result_sha256'] != identity['parent_result_sha256']
            or manifest['guidance_scale'] != identity['guidance_scale']):
        raise ValueError('Require the registered RGB chain and the exact exported endpoint')
    for name, digest in complete['artifact_hashes'].items():
        if Path(name).name != name or sha(reference / name) != digest:
            raise ValueError('Reference artifact changed: ' + name)
    # Always the first preregistered sequence; never choose a successful one.
    sequence = identity['resolved_plan']['rgb_chains'][0]
    rows = [json.loads(line) for line in (reference / 'generations.jsonl').read_text().splitlines()]
    selected = [r for r in rows if (r['sequence'], r['repetition']) == (sequence['sequence'], sequence['repetition'])]
    commands, expected, images = [], [], []
    if len(sequence['steps']) != 6 or len(selected) != 30:
        raise ValueError('Require all six writes and all thirty reads')
    for step in sequence['steps']:
        commands.append({'op': 'write', 'event': step['event_text'], 'seed': step['noise_seed']})
        reads = [r for r in selected if r['step'] == step['step']]
        if len(reads) != 5 or {r['prompt_id'] for r in reads} != set(PROMPTS):
            raise ValueError('Missing or duplicate reference prompt')
        reads = {r['prompt_id']: r for r in reads}
        image_hashes = {r['image_file_sha256'] for r in reads.values()}
        if len(image_hashes) != 1:
            raise ValueError('Reader variants saw different memory images')
        images.append(next(iter(image_hashes)))
        for prompt in PROMPTS:
            row = reads[prompt]
            if row['event_text'] != step['event_text'] or row['noise_seed'] != step['noise_seed']:
                raise ValueError('Reference event or seed differs from the fixed plan')
            if sha(reference / row['image_artifact']) != row['image_file_sha256']:
                raise ValueError('Reference PNG changed')
            commands.append({'op': 'read', 'query': row['query']})
            expected.append({k: row[k] for k in ('query', *FIELDS)})
    output.mkdir(parents=True, exist_ok=False)
    command_file = output / 'commands.jsonl'
    command_file.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in commands), encoding='utf-8')
    expected_path = output / 'expected.json'
    write_json(expected_path, {'reads': expected, 'png_sha256': images})
    write_json(output / 'prepared.json', {
        'reference_complete_sha256': sha(reference / 'complete.json'),
        'reference_interpretation': identity['interpretation'],
        'reference_functional_pass': complete['all_generated_correct_eos'],
        'package_manifest_sha256': sha(package / 'manifest.json'),
        'command_file_sha256': sha(command_file), 'expected_sha256': sha(expected_path),
        'sequence': sequence['sequence'], 'repetition': sequence['repetition'],
        'scope': 'Engineering parity only. This does not certify functional correctness.'})


def verify(prepared, inference):
    prepared, inference = Path(prepared), Path(inference)
    binding = read_json(prepared / 'prepared.json')
    if sha(prepared / 'expected.json') != binding['expected_sha256'] or sha(prepared / 'commands.jsonl') != binding['command_file_sha256']:
        raise ValueError('Prepared replay changed')
    complete = read_json(inference / 'complete.json')
    if (complete['package_manifest_sha256'] != binding['package_manifest_sha256']
            or complete['command_file_sha256'] != binding['command_file_sha256']
            or (complete['commands'], complete['writes'], complete['reads']) != (36, 6, 30)):
        raise ValueError('Inference identity or coverage differs')
    rows = [json.loads(line) for line in (inference / 'results.jsonl').read_text().splitlines()]
    commands = [json.loads(line) for line in (prepared / 'commands.jsonl').read_text().splitlines()]
    if len(rows) != 36 or any(r['index'] != i or r['op'] != commands[i]['op'] for i, r in enumerate(rows)):
        raise ValueError('Missing, duplicated, or reordered inference output')
    expected = read_json(prepared / 'expected.json')
    mismatches = []
    writes = [r for r in rows if r['op'] == 'write']
    for index, (row, digest) in enumerate(zip(writes, expected['png_sha256'], strict=True), 1):
        if row['memory_image'] != f'memory-{index:04d}.png' or sha(inference / row['memory_image']) != row['image_file_sha256']:
            raise ValueError('Inference write artifact changed')
        if row['image_file_sha256'] != digest:
            mismatches.append({'write': index, 'field': 'PNG bytes'})
    for index, (row, target) in enumerate(zip((r for r in rows if r['op'] == 'read'), expected['reads'], strict=True)):
        for field in ('query', *FIELDS):
            if row[field] != target[field]:
                mismatches.append({'read': index, 'field': field})
    if sha(inference / 'memory-final.png') != complete['final_image_sha256'] or complete['final_image_sha256'] != writes[-1]['image_file_sha256']:
        raise ValueError('Final persistent memory differs from the last write')
    result = {**binding, 'inference_complete_sha256': sha(inference / 'complete.json'),
              'inference_results_sha256': sha(inference / 'results.jsonl'),
              'writes_compared': 6, 'reads_compared': 30, 'parity_pass': not mismatches, 'mismatches': mismatches}
    write_json(prepared / 'parity-result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('prepare')
    for name in ('reference', 'package', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--four-gpu-warm-start', action='store_true')
    p = sub.add_parser('verify')
    for name in ('prepared', 'inference'):
        p.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.reference, args.package, args.output, four_gpu_warm_start=args.four_gpu_warm_start)
        return 0
    result = verify(args.prepared, args.inference)
    print(json.dumps(result))
    return 0 if result['parity_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
