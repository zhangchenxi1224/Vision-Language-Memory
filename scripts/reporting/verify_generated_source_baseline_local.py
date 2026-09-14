"""Recount the complete generated-source baseline archive; tensors remain remote."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.reporting.verify_broader_outputs_local import unpack
from scripts.reporting.collect_transition_endpoint import sha, read, jsonl
from scripts.reporting.collect_broader_endpoint import (
    BANK_SHA, GENERATED_SOURCE_COMMIT, CLEAR_RETENTION_COMMIT, phase_summary,
    registered_protocol,
)


def verify(archive, digest):
    with tempfile.TemporaryDirectory(prefix='source-baseline-', dir=ROOT / '.cache') as temp:
        root = Path(temp)
        unpack(archive, digest, root)
        current, reference = root / 'current', root / 'reference'
        bank_path = current / 'bank/manifest.json'
        bank = read(bank_path)
        _, plan, plan_sha = registered_protocol(bank, GENERATED_SOURCE_COMMIT)
        if (sha(bank_path) != BANK_SHA
                or sha(current / 'preregistered-experiment.json') != plan_sha
                or read(current / 'preregistered-experiment.json') != plan):
            raise ValueError('Bank or preregistration differs')
        identity = read(current / 'train/identity.json')
        prior_identity = read(reference / 'train/identity.json')
        result_path = reference / 'train/result.json'
        result = read(result_path)
        initial = identity['initial_writer']
        if (identity['git_commit'] != GENERATED_SOURCE_COMMIT
                or prior_identity['git_commit'] != CLEAR_RETENTION_COMMIT
                or sha(result_path) != plan['reference_result_sha256']
                or result['status'] != 'completed' or result['optimizer_steps'] != 4832
                or initial['parent_commit'] != CLEAR_RETENTION_COMMIT
                or initial['parent_result_sha256'] != sha(result_path)
                or initial['parent_checkpoint_sha256'] != result['checkpoint_sha256']
                or initial['parent_checkpoint_sha256'] != plan['parent_checkpoint_sha256']
                or initial['manifest_sha256'] != plan['initial_package_manifest_sha256']
                or initial['parent_optimizer_steps'] != 4832):
            raise ValueError('Initial parameter lineage differs')
        runtime_sha = sha(current / 'train/runtime.json')
        if runtime_sha != sha(reference / 'train/runtime.json'):
            raise ValueError('Canonical runtime differs from parent')
        gate = read(current / 'train/baseline-reference-check.json')
        if (any(gate.get(key) is not True for key in ('bitwise_latents_and_images',
                'bitwise_trajectories', 'identical_raw_generation_records'))
                or gate['reference_result_sha256'] != sha(result_path)
                or gate['reference_phase'] != 'trained'
                or gate['raw_comparison_exclusion'] != 'phase label only: baseline versus trained'
                or len(gate['samples']) != 302 or len(set(gate['samples'])) != 302):
            raise ValueError('Recorded GPU comparison is incomplete')
        rows, summaries = [], []
        for directory, phase in ((current / 'train/baseline', 'baseline'),
                                 (reference / 'train/trained', 'trained')):
            hashes = read(directory / 'complete.json')['artifact_hashes']
            if sha(directory / 'generations.jsonl') != hashes['generations.jsonl']:
                raise ValueError('Raw record seal differs')
            if set(gate['samples']) != {name for name in hashes if name.endswith('.pt')}:
                raise ValueError('GPU comparison does not cover the complete tensor inventory')
            raw = jsonl(directory / 'generations.jsonl')
            summary, _ = phase_summary(raw, bank, phase)
            if summary['correct_eos'] != 1510 or summary['generated_images'] != 302:
                raise ValueError('Measured baseline differs from parent acceptance')
            rows.append([{k: v for k, v in row.items() if k != 'phase'} for row in raw])
            summaries.append(summary)
        if rows[0] != rows[1]:
            raise ValueError('Raw records differ beyond phase label')
        return {'archive_sha256': digest, 'training_commit': GENERATED_SOURCE_COMMIT,
            'reference_commit': CLEAR_RETENTION_COMMIT, 'plan_sha256': plan_sha,
            'canonical_runtime_sha256': runtime_sha, 'reference_result_sha256': sha(result_path),
            'raw_records_equal_excluding_phase': True, 'recorded_gpu_samples_compared': 302,
            'phases': summaries,
            'scope': 'All 3020 raw records per phase recounted locally, including exact token/EOS scores and complete cell coverage. GPU gate records full tensor/trajectory equality. Tensor payloads and checkpoint are omitted locally and are not rechecked here. This is a baseline, not a trained endpoint.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.archive, args.sha256)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'archive_sha256': result['archive_sha256'],
        'phases': [{k: s[k] for k in ('phase', 'raw_rows', 'correct_eos', 'generated_images')}
                   for s in result['phases']]}))
