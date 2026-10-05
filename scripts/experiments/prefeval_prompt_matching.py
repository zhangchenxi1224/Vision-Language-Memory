"""History-conditioned targets shared by hard distillation and Prompt Matching.

Only the initial, observed exchange is available to the K1 write teacher. No
published answer, correct option, held-out question, or future exchange enters it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path

import torch

from scripts.experiments.prefeval_k1_data import event_text, sha
from vision_memory.reader.open_answer import generate_short_answer
from vision_memory.reader.prompt_matching import (
    qwen3vl_continuation_logits, validate_generated_target_ids,
)
from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT

SUPERVISIONS = ('hard_ce', 'history_hard', 'prompt_matching')
SCHEMA = 'vision_memory.history-prompt-target.v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def history_teacher_query(row, query):
    history = row['history'][:2]
    text = event_text(history)
    if not query.strip():
        raise ValueError('A nonempty training question is required')
    return ('Use the following conversation history to answer the current question.\n'
            '<conversation_history>\n' + text + '\n</conversation_history>\n\n'
            'Current question:\n' + query)


def validate_options(args):
    mode = getattr(args, 'supervision', 'hard_ce')
    if mode not in SUPERVISIONS:
        raise ValueError(f'Unknown supervision: {mode}')
    temperature = getattr(args, 'temperature', 1.0)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if mode == 'history_hard' and temperature != 1.0:
        raise ValueError('history_hard uses hard targets and requires temperature=1')
    if mode == 'hard_ce' and (temperature != 1.0 or getattr(args, 'teacher_cache', None)):
        raise ValueError('Teacher cache/temperature options apply only to history supervision')
    if mode != 'hard_ce' and getattr(args, 'boundary_recovery_from', None):
        raise ValueError('Historical hard-CE recovery cannot resume a different objective')
    if getattr(args, 'teacher_max_new_tokens', 512) <= 0:
        raise ValueError('teacher_max_new_tokens must be positive')
    return mode


def supervision_binding(args, rows):
    """Additional identity only for new modes; legacy hard-CE identity is unchanged."""
    if validate_options(args) == 'hard_ce':
        return {}
    reader_root = Path(args.reader).resolve()
    identity_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                      'preprocessor_config.json', 'chat_template.json',
                      'chat_template.jinja', 'model.safetensors.index.json']
    reader_identity = {name: sha(reader_root / name) for name in identity_files
                       if (reader_root / name).is_file()}
    if not reader_identity:
        raise ValueError('Reader configuration identity files are missing')
    weight_files = sorted({*reader_root.glob('*.safetensors'), *reader_root.glob('pytorch_model*.bin')})
    reader_weights = {path.name: sha(path) for path in weight_files if path.is_file()}
    if not reader_weights:
        raise ValueError('Reader weight files are missing; configuration identity alone is insufficient')
    return {
        'supervision': args.supervision,
        'temperature': args.temperature,
        'teacher_max_new_tokens': args.teacher_max_new_tokens,
        'teacher_source': 'same_frozen_reader_initial_observed_exchange',
        'teacher_reference': 'uniform_128_over_255_rgb_1024',
        'teacher_history_scope': 'history[:2]; no future exchanges or answers',
        'training_history_sha256': digest([(r['base_pair_id'], r['history'][:2]) for r in rows]),
        'reader_path': str(reader_root),
        'reader_config_sha256': reader_identity,
        'reader_weights_sha256': reader_weights,
        'target_pipeline_sha256': sha(Path(__file__)),
        'reader_objective_sha256': sha(Path(__file__).resolve().parents[2] /
                                     'src/vision_memory/reader/prompt_matching.py'),
        'continuation': 'teacher_greedy_exact_token_ids_with_real_assistant_eos',
        'loss': ('full_vocab_soft_cross_entropy_temperature_squared'
                 if args.supervision == 'prompt_matching' else 'teacher_generated_hard_cross_entropy'),
    }


def tensor_digest(value):
    header = json.dumps({'dtype': str(value.dtype), 'shape': list(value.shape)}, sort_keys=True).encode()
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + b'\n' + raw).hexdigest()


def target_cache_binding(binding, row, query):
    # Both modes (and temperatures) reuse exactly the same raw teacher logits.
    return {
        'schema': SCHEMA, 'pair_id': row['base_pair_id'],
        'query': query, 'teacher_query': history_teacher_query(row, query),
        'reader_path': binding['reader_path'],
        'reader_config_sha256': binding['reader_config_sha256'],
        'reader_weights_sha256': binding['reader_weights_sha256'],
        'target_pipeline_sha256': binding['target_pipeline_sha256'],
        'reader_objective_sha256': binding['reader_objective_sha256'],
        'reference': binding['teacher_reference'],
        'max_new_tokens': binding['teacher_max_new_tokens'],
        'generation': 'greedy_no_sampling_no_beam',
    }


def _validate_cached_target(item, key, *, assistant_end_token_id, pad_token_id):
    if item['binding'] != key:
        raise ValueError('Teacher target cache identity mismatch')
    logits = item['logits']
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise ValueError('Cached teacher logits must be a floating tensor')
    if item['logits_sha256'] != tensor_digest(logits):
        raise ValueError('Teacher target cache tensor hash mismatch')
    ids = validate_generated_target_ids(item['generation'], assistant_end_token_id=assistant_end_token_id,
                                        pad_token_id=pad_token_id)
    if list(ids) != item['target_ids']:
        raise ValueError('Cached continuation IDs disagree with generation provenance')
    if logits.ndim != 3 or tuple(logits.shape[:2]) != (1, len(ids)) or logits.shape[-1] == 0:
        raise ValueError('Cached teacher logits are not aligned to continuation tokens')
    if max(ids) >= logits.shape[-1]:
        raise ValueError('Cached continuation token is outside the teacher vocabulary')
    if not torch.isfinite(logits).all():
        raise ValueError('Cached teacher logits are nonfinite')


def get_history_target(*, cache_root, binding, row, query, model, processor, reference,
                       device, assistant_end_token_id):
    """Immutable disk cache of exact IDs and raw logits, shared across paired arms."""
    key = target_cache_binding(binding, row, query)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / (digest(key) + '.pt')
    validation = dict(assistant_end_token_id=assistant_end_token_id,
                      pad_token_id=getattr(processor.tokenizer, 'pad_token_id', None))
    if path.exists():
        item = torch.load(path, map_location='cpu', weights_only=True)
    else:
        with torch.no_grad():
            generation = generate_short_answer(
                model=model, processor=processor, image=reference,
                query=key['teacher_query'], device=device,
                max_new_tokens=binding['teacher_max_new_tokens'], do_sample=False,
                reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
            ids = validate_generated_target_ids(
                generation, assistant_end_token_id=assistant_end_token_id,
                pad_token_id=validation['pad_token_id'])
            output = qwen3vl_continuation_logits(
                model=model, processor=processor, image=reference,
                query=key['teacher_query'], target_ids=torch.tensor([ids], dtype=torch.long),
                device=device, require_image_grad=False,
                reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
        logits = output.target_logits.detach().cpu().contiguous()
        item = {'binding': key, 'generation': generation, 'target_ids': list(ids),
                'logits': logits, 'logits_sha256': tensor_digest(logits)}
        _validate_cached_target(item, key, **validation)
        temp = path.with_suffix(f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
        try:
            torch.save(item, temp)
            try:
                # Link publishes the complete artifact atomically without replacing another writer.
                os.link(temp, path)
            except FileExistsError:
                winner = torch.load(path, map_location='cpu', weights_only=True)
                _validate_cached_target(winner, key, **validation)
                if (winner['target_ids'] != item['target_ids']
                        or winner['logits_sha256'] != item['logits_sha256']):
                    raise ValueError('Concurrent teacher cache writers produced different targets')
                item = winner
        finally:
            temp.unlink(missing_ok=True)
    _validate_cached_target(item, key, **validation)
    return item, path


def validate_teacher_manifest(done, *, arm, supervision='hard_ce', steps=288):
    if supervision not in SUPERVISIONS:
        raise ValueError('Unknown expected teacher supervision')
    if done['step'] != steps or done['binding']['arm'] != arm:
        raise ValueError('Teacher budget or task format does not match Writer')
    actual = done['binding'].get('supervision', 'hard_ce')
    if actual != supervision:
        raise ValueError(f'Teacher objective mismatch: {actual} != {supervision}')
    if actual != 'hard_ce' and not done['binding'].get('reader_objective_sha256'):
        raise ValueError('Soft/history teacher is missing objective provenance')
    return digest(done['binding'])
