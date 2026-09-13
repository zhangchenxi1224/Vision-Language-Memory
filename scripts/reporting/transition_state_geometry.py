"""Compare every sealed transition output with all three teacher states on CPU.

Distances diagnose routing versus readout sensitivity; they never replace raw
answer/EOS correctness and do not establish the cause of a failure on their own.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'bank', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    import torch
    from scripts.reporting.collect_transition_endpoint import BANK_SHA, COMMIT, phase_summary, sha, read, jsonl, seed
    from vision_memory.training.latent_bank_unet import load_teacher_bank
    from vision_memory.repro import canonical_tensor_sha256

    if sha(args.bank) != BANK_SHA:
        raise ValueError('Require the fixed45-condition bank')
    terminal, result = read(args.run / 'terminal.json'), read(args.run / 'train/result.json')
    if terminal.get('state') != 'completed' or terminal['training_result_sha256'] != sha(args.run / 'train/result.json'):
        raise ValueError('Require the completed fixed endpoint')
    identity = read(args.run / 'train/identity.json')
    if identity['git_commit'] != COMMIT or identity['steps'] != 2880 or identity['bank_manifest_sha256'] != BANK_SHA:
        raise ValueError('Unexpected parent training identity')
    bank, teachers = load_teacher_bank(args.bank)
    states = {}
    for group in bank['groups']:
        for teacher_id in group['teacher_ids']:
            value = teachers[teacher_id]
            state = group['target_state']
            if state in states and not torch.equal(states[state], value):
                raise ValueError('Expected one unchanged teacher per state')
            states[state] = value
    if set(states) != {'ambient', 'jazz', 'clear'}:
        raise ValueError('Missing teacher state')
    phase = args.run / 'train/trained'
    complete_sha = sha(phase / 'complete.json')
    complete = read(phase / 'complete.json')
    rows_path = phase / 'generations.jsonl'
    if sha(rows_path) != complete['artifact_hashes']['generations.jsonl']:
        raise ValueError('Raw generation file changed')
    rows = jsonl(rows_path)
    phase_result, _ = phase_summary(rows, bank, 'trained')
    observed = {}
    for row in rows:
        if row['condition'] == 'matched':
            observed.setdefault((row['question_id'], row['noise_seed']), []).append(row)
    images, cells = [], {}
    for group in bank['groups']:
        qid = group['question_id']
        for index in range(4):
            name = hashlib.sha256(qid.encode()).hexdigest()[:16] + f'-seed-{index:02d}.pt'
            path = phase / name
            digest = complete['artifact_hashes'][name]
            if sha(path) != digest:
                raise ValueError('Generated tensor file changed')
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if payload['question_id'] != qid or payload['noise_seed'] != seed(index):
                raise ValueError('Generated tensor identity changed')
            latent = payload['latent']
            if latent.dtype != torch.float32 or tuple(latent.shape) != (1, 4, 128, 128) or not torch.isfinite(latent).all():
                raise ValueError('Invalid generated latent')
            readouts = observed[(qid, seed(index))]
            image_sha = canonical_tensor_sha256(payload['image'])
            if len(readouts) != 5 or any(row['image_sha256'] != image_sha for row in readouts):
                raise ValueError('Stored tensor is not the image read by all five prompts')
            rms = {state: float((latent - teacher).square().mean().sqrt()) for state, teacher in states.items()}
            nearest = min(rms, key=rms.get)
            correct = sum(row['generated_token_ids'] == row['scorer']['gold_token_ids'] + [151645] for row in readouts)
            record = {'question_id': qid, 'source_state': group['source_state'], 'target_state': group['target_state'],
                'operation': group['operation'], 'wording_index': group['wording_index'], 'noise_seed': seed(index),
                'artifact': name, 'artifact_sha256': digest, 'rms_to_teacher_state': rms, 'nearest_state': nearest,
                'correct_eos': correct, 'raw_counts': dict(Counter(row['raw'] for row in readouts))}
            images.append(record)
            cell = cells.setdefault(qid, {'images': 0, 'all_five_correct': 0, 'nearest_states': Counter()})
            cell['images'] += 1
            cell['all_five_correct'] += int(correct == 5)
            cell['nearest_states'][nearest] += 1
            if sha(path) != digest:
                raise ValueError('Generated tensor changed during analysis')
    if len(images) != 180 or sha(phase / 'complete.json') != complete_sha:
        raise ValueError('Incomplete or changed phase')
    failure_images = [row for row in images if row['correct_eos'] != 5]
    summary = {'images': len(images), 'matched_rows': phase_result['matched_rows'],
        'correct_eos': phase_result['correct_eos'], 'failed_images': len(failure_images),
        'failed_images_nearest_expected_teacher': sum(row['nearest_state'] == row['target_state'] for row in failure_images),
        'failed_images_nearest_other_teacher': sum(row['nearest_state'] != row['target_state'] for row in failure_images)}
    output = {'summary': summary, 'parent_commit': COMMIT, 'result_sha256': terminal['training_result_sha256'],
        'checkpoint_sha256': result['checkpoint_sha256'], 'phase_complete_sha256': complete_sha,
        'bank_sha256': BANK_SHA, 'analysis_file_sha256': sha(Path(__file__)),
        'teacher_latent_sha256': {state: canonical_tensor_sha256(value) for state, value in states.items()},
        'cells': cells, 'images': images,
        'scope': 'CPU latent geometry on the complete fixed endpoint. A nearest teacher is a distance diagnostic, not semantic correctness or causal proof.'}
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
