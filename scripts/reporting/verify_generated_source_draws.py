"""Verify the complete source/condition seal and every actual training source choice."""
import hashlib
import json
from pathlib import Path

from scripts.experiments.generated_source_training_protocol import POOL_MANIFEST_SHA, POOL_PLAN_SHA
from scripts.train.generated_source_augmentation import POLICY, GENERATION_COMMIT, source_variant_index
from scripts.experiments.broader_writer_protocol import SEED

POOL_PATH = '/inspire/ssd/project/exploration-topic/czxs26210936/runs/dreamlite-official-alignment/90b41a2-generated-source-pool/manifest.json'


def pool_manifest():
    path = Path(__file__).resolve().parents[2] / 'reports/official-alignment-results-20260913/90b41a2-generated-source-pool-manifest.json'
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != POOL_MANIFEST_SHA:
        raise ValueError('Changed independently sealed complete source pool')
    return json.loads(data)


def identity_binding():
    return {'policy': POLICY, 'manifest': POOL_PATH, 'manifest_sha256': POOL_MANIFEST_SHA,
        'generation_commit': GENERATION_COMMIT, 'plan_sha256': POOL_PLAN_SHA,
        'choices': 'One original canonical source plus all eight independent generated sources of the same source state.',
        'selection': 'Per conditional question, balanced seeded nine-choice cycles over its logical31/nine-expression occurrences.',
        'evaluation': 'Original canonical source and event condition, unchanged.', 'writer_metadata_input': False}


def verify_seal(bank, identity, runtime, seal, registered):
    if identity.get('generated_source_pool') != identity_binding():
        raise ValueError('Wrong generated source pool training identity')
    registered_pool = registered['generated_source_pool']
    if (registered_pool['manifest_sha256'] != POOL_MANIFEST_SHA or registered_pool['plan_sha256'] != POOL_PLAN_SHA
            or registered_pool['generation_commit'] != GENERATION_COMMIT or registered_pool['policy'] != POLICY):
        raise ValueError('Wrong registered source augmentation')
    groups = {g['question_id']: g for g in bank['groups'] if g.get('source_kind') == 'sealed_rgb_1024'}
    if len(groups) != 108 or set(seal) != set(groups):
        raise ValueError('Require all108 source conditions and no gray/historical replacement')
    pool = pool_manifest()
    source_sets = {state: sorted((r for r in pool['records'] if r['state'] == state), key=lambda r: r['job'])
                   for state in ('ambient', 'jazz', 'clear')}
    for qid, group in groups.items():
        entries = seal[qid]
        records = source_sets[group['source_state']]
        if len(entries) != 9 or len(records) != 8:
            raise ValueError('Require canonical plus all eight generated sources')
        for index, entry in enumerate(entries):
            if (set(entry) != {'index', 'source_image_file_sha256', 'source_latent_sha256', 'event_text_sha256',
                              'prompt_embeds_sha256', 'attention_mask_sha256'} or entry['index'] != index
                    or entry['event_text_sha256'] != hashlib.sha256(group['event_text'].encode()).hexdigest()):
                raise ValueError('Source condition uses a changed event or index')
            expected_png = group['source_image_file_sha256'] if index == 0 else records[index-1]['png_sha256']
            expected_latent = group['source_latent_sha256'] if index == 0 else records[index-1]['source_latent_sha256']
            if (entry['source_image_file_sha256'] != expected_png or entry['source_latent_sha256'] != expected_latent):
                raise ValueError('Source image/latent pair differs from the sealed canonical or generated source')
            for name in ('prompt_embeds_sha256', 'attention_mask_sha256'):
                value = entry[name]
                if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError('Malformed actual source condition tensor hash')
        if entries[0]['prompt_embeds_sha256'] != runtime['condition_sha256'][qid]:
            raise ValueError('Canonical evaluation condition changed')
    return {'conditions': len(groups), 'source_condition_pairs': 108 * 9, 'pool_images': 24,
        'pool_manifest_sha256': POOL_MANIFEST_SHA,
        'scope': 'All recorded source PNG/latent and event/embedding/mask bindings checked against the exact fixed pool and canonical runtime. Native condition tensors are bound to executed training metadata; this CPU collector does not re-encode embeddings.'}


def verify_draw(group, draw, draw_index, seal):
    if group.get('source_kind') == 'sealed_rgb_1024':
        qid = group['question_id']
        index = source_variant_index(SEED, draw_index, qid)
        if draw.get('training_source_variant') != seal[qid][index]:
            raise ValueError('Actual draw used another source PNG, latent or event encoding')
        return qid, str(index)
    if 'training_source_variant' in draw:
        raise ValueError('Unexpected gray/historical source augmentation')
    return None
