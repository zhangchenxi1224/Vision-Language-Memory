"""Seal all64 old EOS endpoints for later FP32/RGB readback, without selecting successes."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

PANEL_SHA = 'd356238fd5c267812dcf28d214ab062fd43388bb6b53b78602f0c1e8f5b36672'
AUDIT_SHA = 'c1a9706a7eb3fa2ee9d3e25486f07ac738e45017d032cbffd96ee616ec21a460'
OLD_COMMIT = '2c0e41c899910bf0641f16ee724f85bbe3491a7e'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def target_spec(target):
    query = target['inputs']['original_open']
    match = re.search(r'current ' + re.escape(target['topic']) + r' preference for ([^?]+)\?', query)
    if not match:
        raise ValueError('Cannot bind the entity in the original question')
    entity = match.group(1)
    events = []
    for record in target['source_prefix']:
        if record['type'] in ('event', 'mixed'):
            # A mixed record contributes its event, never its attached question.
            events.append({key: record[key] for key in ('event_kind', 'event_text')})
    if not events or any(not event['event_text'].strip() for event in events):
        raise ValueError('Missing original event stream')
    questions = {'original_open': query, 'paraphrase_1': target['inputs']['paraphrase_open'],
                 'paraphrase_2': target['inputs']['new_rewrite_open'],
                 'paraphrase_3': f"Which {target['topic']} preference is currently stored for {entity}?\nUse the memory image to answer.\nAnswer with a short phrase only.",
                 'paraphrase_4': f"Read the memory image and report the current {target['topic']} preference of {entity}.\nReturn only a short phrase."}
    if len(set(questions.values())) != 5:
        raise ValueError('Readback questions must be distinct')
    return {'target_index': target['target_index'], 'semantic_group_id': target['semantic_group_id'],
            'entity': entity, 'topic': target['topic'], 'stratum': target['stratum'],
            'gold': target['scorer_metadata']['gold'], 'question_variants': questions,
            'event_stream': events, 'event_stream_scope': 'future Writer input only; never supplied to Reader',
            'question_exposure': 'original_open trained historically; paraphrase_1/2 previously evaluated; paraphrase_3/4 new for this diagnostic'}


def prepare(panel_path, audit_path, campaign, output):
    panel_path, audit_path, campaign, output = map(Path, (panel_path, audit_path, campaign, output))
    if sha(panel_path) != PANEL_SHA or sha(audit_path) != AUDIT_SHA:
        raise ValueError('Historical panel or completed audit changed')
    panel, audit = read(panel_path), read(audit_path)
    targets = [target_spec(t) for t in panel['targets']]
    if len(targets) != 16 or len({t['semantic_group_id'] for t in targets}) != 16:
        raise ValueError('Require the original sixteen distinct questions')
    members = []
    for record in audit['runs']:
        run = Path(record['run'])
        match = re.fullmatch(r'target-(\d+)-seed-(\d+)-B', run.name)
        if match is None:
            continue
        if not run.resolve().is_relative_to(campaign.resolve()):
            raise ValueError('Run lies outside the bound historical campaign')
        if sha(run / 'artifact_inventory.json') != record['inventory_sha256']:
            raise ValueError('Previously audited inventory changed')
        inventory = {item['path']: item for item in read(run / 'artifact_inventory.json')['files']}
        for name in ('checkpoint_index.jsonl', 'manifest.json', 'terminal.json', 'checkpoints/step-256.pt'):
            if sha(run / name) != inventory[name]['sha256']:
                raise ValueError('Historical endpoint binding changed: ' + name)
        index = [json.loads(line) for line in (run / 'checkpoint_index.jsonl').read_text().splitlines()]
        endpoint = [item for item in index if item['step'] == 256]
        if len(endpoint) != 1 or endpoint[0]['path'] != 'checkpoints/step-256.pt':
            raise ValueError('Missing fixed endpoint')
        binding = read(run / 'manifest.json')
        target, seed = map(int, match.groups())
        if (binding['target_index'], binding['seed'], binding['arm']) != (target, seed, 'B') or not binding['append_eos']:
            raise ValueError('Wrong historical EOS arm')
        if read(run / 'terminal.json')['status'] != 'completed':
            raise ValueError('Incomplete old run')
        members.append({'target_index': target, 'seed': seed, 'historical_run': str(run),
            'inventory_sha256': record['inventory_sha256'], 'checkpoint': str(run / 'checkpoints/step-256.pt'),
            'checkpoint_sha256': inventory['checkpoints/step-256.pt']['sha256'],
            'latent_sha256': endpoint[0]['latent_sha256']})
    if len(members) != 64 or {(m['target_index'], m['seed']) for m in members} != {(i, s) for i in range(16) for s in range(4)}:
        raise ValueError('Require every original target and seed; no successful-target substitution')
    result = {'status': 'preregistered_readback_panel_not_teacher_bank', 'historical_commit': OLD_COMMIT,
              'historical_panel_sha256': PANEL_SHA, 'historical_audit_sha256': AUDIT_SHA,
              'targets': targets, 'members': members, 'optimizer_updates': 0,
              'planned_image_forms': ['fp32_vae_decoded', 'rgb_uint8'], 'planned_raw_rows': 640,
              'planned_blank_control_rows': 80, 'planned_blank_image': 'one RGB1024x1024 image filled with (128,128,128), repeated across questions',
              'endpoint_step': 256, 'selection': 'all16 targets x all4 seeds, arm B; no outcome selection',
              'topic_counts': dict(Counter(t['topic'] for t in targets)),
              'scope': 'Future same-snapshot FP32/RGB oracle readback only; neither teacher-bank qualification nor Writer success.'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('panel', 'audit', 'campaign', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    result = prepare(a.panel, a.audit, a.campaign, a.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('targets', 'members')}))
