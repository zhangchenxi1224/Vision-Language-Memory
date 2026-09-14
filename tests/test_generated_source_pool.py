import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from scripts.experiments.generated_source_pool_protocol import plan, digest, PACKAGE_SHA, seed
from scripts.probes.generate_training_source_pool import collect, write
from vision_memory.repro import canonical_tensor_sha256


def test_source_plan_is_independent_training_material():
    actual = plan()
    committed = Path(__file__).resolve().parents[1] / 'reports/official-alignment-results-20260913/generated-training-source-pool-preregistered.json'
    assert actual == json.loads(committed.read_bytes())
    assert len(actual['jobs']) == 24
    assert len({job['seed'] for job in actual['jobs']}) == 24
    assert {job['seed'] for job in actual['jobs']}.isdisjoint(seed('training-noise', i) for i in range(19328))
    assert [sum(j['lane'] == lane for j in actual['jobs']) for lane in range(4)] == [6] * 4
    assert all(len(job['queries']) == 5 for job in actual['jobs'])
    assert actual['protocol']['optimization_steps'] == 0


@pytest.fixture
def packet(tmp_path):
    registered = plan()
    args = SimpleNamespace(output=tmp_path / 'output', plan=tmp_path / 'plan.json', expected_commit='fixed-probe')
    args.output.mkdir()
    write(args.plan, registered)
    for lane in range(4):
        directory = args.output / f'lane-{lane}'
        directory.mkdir()
        records, rows = [], []
        for index, job in enumerate(j for j in registered['jobs'] if j['lane'] == lane):
            png, pt = directory / (job['id'] + '.png'), directory / (job['id'] + '.pt')
            Image.new('RGB', (1024, 1024), (128 + index, 128, 128)).save(png)
            noise = torch.randn((1, 4, 128, 128), generator=torch.Generator().manual_seed(job['seed']))
            latent = torch.zeros_like(noise)
            torch.save({'source_latent': latent, 'generation_initial_source': latent, 'noise': noise,
                        'generated_latent': latent, 'trajectory': (noise,) * 28 + (latent,)}, pt)
            records.append({'job': job['id'], 'state': job['state'], 'seed': job['seed'], 'png': png.name,
                'png_sha256': digest(png), 'tensor': pt.name, 'tensor_sha256': digest(pt),
                'source_latent_sha256': canonical_tensor_sha256(latent), 'correct_reads': 5, 'reads': 5})
            rows += [{'job': job['id'], 'prompt_id': prompt_id, 'query': query,
                      'generated_token_ids': job['expected_token_ids'], 'correct_eos': True}
                     for prompt_id, query in job['queries'].items()]
        (directory / 'reads.jsonl').write_bytes(b'\n'.join(json.dumps(row).encode() for row in rows) + b'\n')
        write(directory / 'complete.json', {'commit': args.expected_commit, 'plan_sha256': digest(args.plan),
            'package_sha256': PACKAGE_SHA, 'reads_sha256': digest(directory / 'reads.jsonl'), 'records': records})
    return args, registered


def test_one_failed_read_rejects_entire_pool_without_dropping_images(packet):
    args, registered = packet
    assert collect(args, registered)['qualified_for_training'] is True
    directory = args.output / 'lane-0'
    rows = [json.loads(line) for line in (directory / 'reads.jsonl').read_bytes().splitlines()]
    rows[0]['generated_token_ids'] = [2152, 4541, 21933, 151645]
    rows[0]['correct_eos'] = False
    (directory / 'reads.jsonl').write_bytes(b'\n'.join(json.dumps(row).encode() for row in rows) + b'\n')
    complete = json.loads((directory / 'complete.json').read_bytes())
    complete['reads_sha256'] = digest(directory / 'reads.jsonl')
    complete['records'][0]['correct_reads'] = 4
    write(directory / 'complete.json', complete)
    result = collect(args, registered)
    assert not result['qualified_for_training']
    assert result['correct_reads'] == 119 and result['raw_reads'] == 120
    assert len(result['records']) == result['generated_images'] == 24


@pytest.mark.parametrize('mutation', ['missing_case', 'query', 'noise', 'immediate_eos'])
def test_source_collector_rejects_bad_evidence(packet, mutation):
    args, registered = packet
    directory = args.output / 'lane-0'
    complete = json.loads((directory / 'complete.json').read_bytes())
    if mutation == 'missing_case':
        complete['records'].pop()
    elif mutation == 'noise':
        path = directory / complete['records'][0]['tensor']
        payload = torch.load(path, weights_only=True)
        payload['noise'] = payload['noise'] + 0.01
        payload['trajectory'] = (payload['noise'],) + payload['trajectory'][1:]
        torch.save(payload, path)
        complete['records'][0]['tensor_sha256'] = digest(path)
    else:
        rows = [json.loads(line) for line in (directory / 'reads.jsonl').read_bytes().splitlines()]
        if mutation == 'query':
            rows[0]['query'] = 'substituted query'
        else:
            rows[0]['generated_token_ids'] = rows[0]['generated_token_ids'][:-1]
        (directory / 'reads.jsonl').write_bytes(b'\n'.join(json.dumps(row).encode() for row in rows) + b'\n')
        complete['reads_sha256'] = digest(directory / 'reads.jsonl')
    write(directory / 'complete.json', complete)
    with pytest.raises(ValueError):
        collect(args, registered)
