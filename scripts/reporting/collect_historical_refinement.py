"""Recount both fixed teacher refinements, retaining all failed raw outputs and step records.

This standard-library collector can run on the CPU transfer instance. It verifies
all remote artifact hashes; the portable archive keeps every text record and the
two endpoint PNGs per target, with large tensor/intermediate PNG omissions explicit.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tarfile

BANK_SHA = 'd54895adb15b91c3befd52c57f6248c76cf2abadb44916af830726a0736e688c'
ARMS = {
    'three': {'commit': '685d65792b3c63e199a4558800113cfbfbe0d712',
        'complete': '7b25d29416a3330388e7fee534f97a58c37cbc39f9334545e04e3fb348e5c33c',
        'plan': 'b1fcd51f6084540f7686391b4753b8b7dfa10574be58db8cc38ca69cac2a495b'},
    'five': {'commit': '918e7d21ded6bff05133aa856f445ae6a4c089a0',
        'complete': 'e52d4777e99ea6f2dbf8614ce0b0acd2afc2e305382adb2fd781bd596179688d',
        'plan': '939d1eacd7e3040f5ed06dfe3c48c770385b8cd538dcc5663ff9a91ed84f172c'}}
GOLD_IDS = {'green': [13250], 'juice': [8613, 558], 'jazz': [73, 9802], 'linen': [3732, 268],
            'pasta': [79, 14300], 'no active preference': [2152, 4541, 21933]}
CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 256)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines()]


def strict(row, gold):
    if row['scorer']['gold_token_ids'] != GOLD_IDS[gold]:
        raise ValueError('Actual Reader gold tokenization changed')
    passed = row['generated_token_ids'] == GOLD_IDS[gold] + [151645]
    if passed != bool(row['scorer']['strict_correct'] and row['scorer']['answer_followed_immediately_by_eos']):
        raise ValueError('Raw tokens disagree with strict immediate-EOS score')
    return passed


def portable(path):
    return path.suffix in ('.json', '.jsonl', '.log') or path.name in ('step-256.png', 'endpoint.png')


def collect(run, parent_bank, arm, *, text_only=False):
    run, parent_bank = Path(run), Path(parent_bank)
    fixed = ARMS[arm]
    if sha(parent_bank) != BANK_SHA or sha(run / 'complete.json') != fixed['complete'] or sha(run / 'preregistered-plan.json') != fixed['plan']:
        raise ValueError('Original bank, fixed complete record or preregistration changed')
    bank, plan, done = read(parent_bank), read(run / 'preregistered-plan.json'), read(run / 'complete.json')
    groups = sorted(bank['groups'], key=lambda group: group['historical_target_index'])
    teachers = {teacher['teacher_id']: teacher for teacher in bank['teachers']}
    if len(teachers) != 64 or [group['historical_target_index'] for group in groups] != list(range(16)):
        raise ValueError('Incomplete original16-question/64-target population')
    if read(run / 'status.json')['complete_sha256'] != fixed['complete'] or done['state'] != 'completed':
        raise ValueError('Not the completed fixed experiment')
    results, omitted, verified = {}, [], 0
    for group, selected in zip(groups, plan['selected'], strict=True):
        index = group['historical_target_index']
        tid = min(group['teacher_ids'], key=lambda value: hashlib.sha256(value.encode()).hexdigest())
        expected_selection = {'target_index': index, 'question_id': group['question_id'], 'teacher_id': tid,
            'latent_sha256': teachers[tid]['latent_sha256'], 'historical_checkpoint_sha256': teachers[tid]['historical_checkpoint_sha256']}
        if selected != expected_selection:
            raise ValueError('Original target was replaced after observing outcomes')
        lane = run / 'lanes' / f'lane-{index % 2}'
        lane_identity, lane_done = read(lane / 'identity.json'), read(lane / 'complete.json')
        if lane_identity['commit'] != fixed['commit'] or lane_identity['plan_sha256'] != fixed['plan'] or lane_done['state'] != 'completed':
            raise ValueError('Lane source or plan binding differs')
        directory = lane / 'runs' / f'target-{index:02d}'
        qualified = read(directory / 'qualification.json')
        if (qualified['selected'] != selected
                or sha(directory / 'qualification.json') != lane_done['results'][str(index)]['qualification_sha256']):
            raise ValueError('Target qualification identity changed')
        for name, digest in qualified['artifact_hashes'].items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError('Invalid artifact path')
            if text_only and not portable(path) and not path.exists():
                omitted.append(path.relative_to(run.resolve()).as_posix())
            elif sha(path) != digest:
                raise ValueError('Changed refinement artifact: ' + name)
            else:
                verified += 1
        terminal, manifest = read(directory / 'terminal.json'), read(directory / 'manifest.json')
        if (terminal['status'] != 'completed' or terminal['optimizer_steps'] != 256 or terminal['latent_count'] != 257
                or terminal['checkpoint_count'] != 11 or terminal['latent_sha256'] != selected['latent_sha256']
                or manifest['initial_latent_sha256'] != selected['latent_sha256'] or manifest['training_prompts'] != plan['training_prompts']):
            raise ValueError('Initialization, prompt exposure or fixed endpoint differs')
        gradient = read(directory / 'initial_reproducibility.json')
        if not gradient['loss_equal'] or not gradient['gradient_equal'] or not math.isfinite(gradient['gradient_rms']) or gradient['gradient_rms'] <= 0:
            raise ValueError('Actual latent gradient preflight failed')
        metrics = jsonl(directory / 'metrics.jsonl')
        prompts = plan['training_prompts']
        if len(metrics) != 256:
            raise ValueError('Missing latent optimizer records')
        for step, metric in enumerate(metrics):
            if metric['optimizer_step'] != step + 1 or metric['training_prompt_id'] != prompts[step % len(prompts)]:
                raise ValueError('Actual round-robin training query exposure changed')
            if any(not math.isfinite(metric[key]) for key in ('loss_before_step', 'gradient_rms', 'update_rms')):
                raise ValueError('Nonfinite latent training record')
        latents = jsonl(directory / 'latent_index.jsonl')
        checkpoints = jsonl(directory / 'checkpoint_index.jsonl')
        if [row['optimizer_step'] for row in latents] != list(range(257)) or [row['optimizer_step'] for row in checkpoints] != list(CHECKPOINTS):
            raise ValueError('Incomplete saved trajectory or optimizer checkpoints')
        for item in latents + checkpoints:
            if item['file_sha256'] != qualified['artifact_hashes'][item['path']]:
                raise ValueError('Trajectory index differs from the sealed file')
        if latents[0]['latent_sha256'] != selected['latent_sha256'] or latents[-1]['latent_sha256'] != terminal['endpoint_latent_sha256']:
            raise ValueError('Saved trajectory initial/final latent binding differs')
        checkpoint_rows = jsonl(directory / 'checkpoint_generations.jsonl')
        if [row['optimizer_step'] for row in checkpoint_rows] != list(CHECKPOINTS):
            raise ValueError('Missing checkpoint raw readback')
        for row in checkpoint_rows:
            if row['query'] != group['question_variants']['original_open'] or row['condition'] != 'matched':
                raise ValueError('Checkpoint query changed')
            strict(row, group['answer'])
        rows = jsonl(directory / 'generations.jsonl') + jsonl(directory / 'rgb/generations.jsonl')
        expected = {(form, condition, prompt) for form in ('fp32_vae_decoded', 'rgb_uint8')
            for condition in ('matched', 'blank', 'fixed_donor') for prompt in group['question_variants']}
        seen, images, passed, failures = set(), {}, 0, []
        for row in rows:
            key = row['image_form'], row['condition'], row['prompt_id']
            if (key not in expected or key in seen or row['target_index'] != index or row['gold'] != group['answer']
                    or row['optimizer_step'] != 256 or row['query'] != group['question_variants'][key[2]]):
                raise ValueError('Unexpected, repeated or changed endpoint readback')
            seen.add(key)
            if images.setdefault(key[:2], row['image_sha256']) != row['image_sha256']:
                raise ValueError('Questions read different images')
            correct = strict(row, group['answer'])
            if row['condition'] == 'matched':
                passed += int(correct)
                if not correct:
                    failures.append({'form': key[0], 'prompt': key[2], 'raw': row['raw'], 'tokens': row['generated_token_ids']})
        if seen != expected or len(rows) != 30:
            raise ValueError('Incomplete endpoint matrix')
        summary = {'raw_rows': 30, 'matched_rows': 10, 'correct_eos': passed, 'both_forms_all_five': passed == 10, 'writer_success': False}
        if qualified['summary'] != summary or done['results'][str(index)] != summary:
            raise ValueError('Strict raw outcome differs from the qualified result')
        results[str(index)] = {**summary, 'failures': failures, 'question_id': group['question_id'],
            'gold': group['answer'], 'training_prompt_counts': Counter(row['training_prompt_id'] for row in metrics),
            'qualified_sha256': sha(directory / 'qualification.json')}
    return {'arm': arm, 'source_commit': fixed['commit'], 'complete_sha256': fixed['complete'], 'plan_sha256': fixed['plan'],
        'targets': 16, 'raw_rows': 480, 'matched_rows': 160, 'correct_eos': sum(value['correct_eos'] for value in results.values()),
        'all_ten_targets': sum(value['both_forms_all_five'] for value in results.values()),
        'additional_latent_updates': 4096, 'trajectory_records': 4112, 'checkpoint_raw_rows': 176,
        'verified_artifact_files': verified, 'artifacts_omitted_locally': omitted, 'all_artifacts_verified_here': not omitted,
        'scope': 'Teacher refinement only; all queries already diagnostic-seen; not shared Writer success.', 'results': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'parent-bank', 'output-prefix'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=('three', 'five'), required=True)
    parser.add_argument('--text-only', action='store_true')
    a = parser.parse_args()
    summary = collect(a.run, a.parent_bank, a.arm, text_only=a.text_only)
    output = Path(str(a.output_prefix) + '-summary.json')
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    if not a.text_only:
        archive_path = Path(str(a.output_prefix) + '-evidence.tgz')
        with tarfile.open(archive_path, 'w:gz') as archive:
            for path in sorted(a.run.rglob('*')):
                if path.is_file() and portable(path):
                    archive.add(path, arcname=path.relative_to(a.run).as_posix())
            archive.add(output, arcname='verified-summary.json')
        summary['archive_sha256'] = sha(archive_path)
        summary['archive_bytes'] = archive_path.stat().st_size
    print(json.dumps({key: value for key, value in summary.items() if key not in ('results', 'artifacts_omitted_locally')}))
