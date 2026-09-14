"""Training-only source PNG variation; canonical evaluation contexts stay intact."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.experiments.generated_source_pool_protocol import plan, digest, BANK_SHA
from vision_memory.training.latent_bank_unet import stable_seed

GENERATION_COMMIT = '90b41a2f1e7e4e0a8d41e709cedb69c4b32e3231'
POLICY = 'canonical-plus-eight-generated-source-pngs/v1'


def source_pool_binding(args):
    path = getattr(args, 'generated_source_pool', None)
    expected = getattr(args, 'generated_source_pool_sha256', None)
    if path is None and expected is None:
        return None
    if not path or not expected:
        raise ValueError('Generated source pool requires an explicit manifest SHA256')
    if (args.model_variant != 'base' or args.flow_protocol != 'official' or args.prompt_style != 'native_base'
            or args.sampling_strategy != 'logical_condition' or args.seed != 20260915
            or digest(args.bank_manifest) != BANK_SHA):
        raise ValueError('Source variants require the fixed151 bank, logical31 sampling, seed20260915 and official native Base')
    path = Path(path).resolve()
    if digest(path) != expected:
        raise ValueError('Changed generated source pool manifest')
    manifest = json.loads(path.read_bytes())
    registered = plan()
    if (manifest.get('commit') != GENERATION_COMMIT or manifest.get('schema') != registered['schema']
            or manifest.get('parent_package_sha256') != registered['parent_package_sha256']
            or manifest.get('qualified_for_training') is not True or manifest.get('generated_images') != 24
            or manifest.get('raw_reads') != 120 or manifest.get('correct_reads') != 120
            or digest(path.parent / 'plan.json') != manifest.get('plan_sha256')
            or json.loads((path.parent / 'plan.json').read_bytes()) != registered):
        raise ValueError('Require the complete qualified, fixed generated-source pool')
    return {'policy': POLICY, 'manifest': str(path), 'manifest_sha256': expected,
        'generation_commit': GENERATION_COMMIT, 'plan_sha256': manifest['plan_sha256'],
        'choices': 'One original canonical source plus all eight independent generated sources of the same source state.',
        'selection': 'Per conditional question, balanced seeded nine-choice cycles over its logical31/nine-expression occurrences.',
        'evaluation': 'Original canonical source and event condition, unchanged.', 'writer_metadata_input': False}


def source_variant_index(seed, draw_index, question_id):
    if seed != 20260915 or draw_index < 0:
        raise ValueError('Require the registered seed and a nonnegative draw index')
    # In the fixed151 bank each music expression occurs once per nine logical31
    # cycles. A full nine-choice cycle therefore covers each image equally.
    occurrence = (draw_index // 31) // 9
    cycle, offset = divmod(occurrence, 9)
    generator = torch.Generator().manual_seed(stable_seed(seed, 'training-source-order:' + question_id, cycle))
    return int(torch.randperm(9, generator=generator)[offset])


@torch.no_grad()
def install_source_variants(args, bank, pipe, contexts, source_image_files):
    """Verify all generation records, then encode the actual selected PNG/event pairs."""
    from PIL import Image
    from scripts.probes.generate_training_source_pool import collect
    from vision_memory.repro import canonical_tensor_sha256
    from vision_memory.dreamlite.conditioning import encode_native_base_edit_condition
    binding = source_pool_binding(args)
    if binding is None:
        return {}
    manifest_path = Path(binding['manifest'])
    pool = manifest_path.parent
    manifest = json.loads(manifest_path.read_bytes())
    # Recount all120 original raw reads, all24 tensors and PNGs. The current
    # training source commit can differ from the sealed generation commit.
    verified = collect(SimpleNamespace(output=pool, plan=pool / 'plan.json', expected_commit=GENERATION_COMMIT), plan())
    if verified != manifest:
        raise ValueError('Complete source pool does not replay from actual artifacts')
    vd = torch.device(args.dreamlite_device)
    generated = {}
    for state in ('ambient', 'jazz', 'clear'):
        records = sorted((r for r in manifest['records'] if r['state'] == state), key=lambda r: r['job'])
        if [r['job'] for r in records] != [f'{state}-{i:02d}' for i in range(8)]:
            raise ValueError('Missing or substituted source image')
        generated[state] = []
        for record in records:
            path = pool / record['png']
            with Image.open(path) as image:
                image = image.copy()
            source = pipe.prepare_image_latents(pipe.image_processor.preprocess(image), dtype=torch.float32, device=vd)
            saved = torch.load(pool / record['tensor'], map_location='cpu', weights_only=True)['source_latent']
            if not torch.equal(source.cpu(), saved):
                raise ValueError('Generated PNG source differs from official VAE encoding in the training runtime')
            source_image_files[path] = record['png_sha256']
            generated[state].append((image, source, record))
    encoded_binding = {}
    for group in bank['groups']:
        if group.get('source_state') not in generated:
            continue
        if group.get('source_kind') != 'sealed_rgb_1024' or not 0 <= group.get('wording_index', -1) < 9:
            raise ValueError('Generated sources apply only to the complete original music source/wording strata')
        qid = group['question_id']
        context = contexts[qid]
        choices = [(context['source'], context['condition'], group['source_image_file_sha256'])]
        for image, source, record in generated[group['source_state']]:
            condition = encode_native_base_edit_condition(pipe, image, group['event_text'], device=vd, dtype=torch.float32)
            choices.append((source, condition, record['png_sha256']))
        bindings = []
        for index, (source, condition, png_sha) in enumerate(choices):
            if (source.dtype != torch.float32 or not torch.isfinite(source).all()
                    or condition.prompt_embeds.dtype != torch.float32 or not torch.isfinite(condition.prompt_embeds).all()):
                raise ValueError('Nonfinite or non-FP32 generated source condition')
            bindings.append({'index': index, 'source_image_file_sha256': png_sha,
                'source_latent_sha256': canonical_tensor_sha256(source),
                'event_text_sha256': hashlib.sha256(group['event_text'].encode()).hexdigest(),
                'prompt_embeds_sha256': canonical_tensor_sha256(condition.prompt_embeds),
                'attention_mask_sha256': canonical_tensor_sha256(condition.attention_mask)})
        context['training_source_variants'] = [(source, condition) for source, condition, _ in choices]
        context['training_source_variant_binding'] = bindings
        encoded_binding[qid] = bindings
    if len(encoded_binding) != 108:
        raise ValueError('Require all108 sealed music-source conditions; no gray/historical replacement')
    if source_pool_binding(args) != binding:
        raise ValueError('Source pool changed during condition preparation')
    return encoded_binding
