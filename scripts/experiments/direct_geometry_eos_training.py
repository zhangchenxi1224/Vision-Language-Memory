"""Fresh direct-z trajectories with answer + assistant-termination supervision."""
from __future__ import annotations
import json
import time
from pathlib import Path
import torch
from scripts.experiments import run_r11_mcq_open_multistart as storage
from scripts.experiments import run_r11_open_answer_replay as replay
from scripts.train.latent_r11_vae_oracle import VAELatentOracle
from vision_memory.reader.open_answer import generate_short_answer
from vision_memory.reader.open_eos import qwen3vl_answer_eos_ce, generation_diagnostics
from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT
from vision_memory.repro import canonical_tensor_sha256, configure_strict_cuda_determinism
from vision_memory.training.direct_latent_geometry import CHECKPOINTS


def teacher(runtime, image, prompt, require_grad=False):
    return qwen3vl_answer_eos_ce(
        model=runtime['reader'], processor=runtime['processor'],
        image=image[0].to(runtime['reader_device']), device=runtime['reader_device'],
        query=runtime['target']['inputs'][prompt], target=runtime['target']['scorer_metadata']['gold'],
        termination=runtime['termination'], lambda_eos=1.0,
        require_image_grad=require_grad, reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,
        deterministic_ce=True)


def gradient_gate(oracle, runtime):
    losses, gradients = [], []
    with storage.preserve_rng():
        state = storage.capture_rng()
        for _ in range(2):
            storage.restore_rng(state)
            oracle.latent_fp32.grad = None
            output = teacher(runtime, oracle.image(), 'original_open', True)
            output.loss.backward()
            gradient = oracle.latent_fp32.grad
            if (gradient is None or not torch.isfinite(output.loss) or not torch.isfinite(gradient).all()
                    or not bool((gradient != 0).any())):
                raise RuntimeError('EOS objective has no finite, nonzero latent gradient')
            losses.append(output.loss.detach().cpu().clone())
            gradients.append(gradient.detach().cpu().clone())
            del output
        oracle.latent_fp32.grad = None
    if not torch.equal(*losses) or not torch.equal(*gradients):
        raise RuntimeError('Repeated EOS loss/gradient is not bitwise deterministic')
    return {'loss_equal': True, 'gradient_equal': True, 'loss': float(losses[0]),
            'gradient_sha256': canonical_tensor_sha256(gradients[0]),
            'gradient_rms': float(gradients[0].square().mean().sqrt())}


@torch.no_grad()
def evaluate(runtime, image, *, spec, step, directory, all_prompts):
    prompts = runtime['target']['inputs'] if all_prompts else {'original_open': runtime['target']['inputs']['original_open']}
    conditions = {'matched': image, **runtime['controls']} if all_prompts else {'matched': image}
    output_rows = []
    with storage.preserve_rng():
        for condition, pixels in conditions.items():
            for prompt_id, query in prompts.items():
                output = teacher(runtime, pixels, prompt_id)
                gold_ids = output.target_ids[0, :output.answer_token_count].cpu().tolist()
                generated = generate_short_answer(model=runtime['reader'], processor=runtime['processor'],
                    image=pixels.to(runtime['reader_device']), query=query, device=runtime['reader_device'],
                    max_new_tokens=32, do_sample=False)
                row = {**spec, 'optimizer_step': step, 'condition': condition, 'prompt_id': prompt_id,
                    'query': query, 'question_trained': prompt_id == 'original_open',
                    'image_sha256': canonical_tensor_sha256(pixels), **generated,
                    'answer_ce': float(output.answer_loss), 'eos_ce': float(output.eos_loss),
                    'total_ce': float(output.loss), 'gold_eos_appended': True,
                    'scorer': generation_diagnostics(generated, runtime['target']['scorer_metadata']['gold'], gold_ids)}
                replay.append_jsonl(directory / ('generations.jsonl' if all_prompts else 'checkpoint_generations.jsonl'), row)
                output_rows.append(row)
    return output_rows


def run_one(*, spec, initial, reference, runtime, output_dir):
    directory = output_dir / 'runs' / spec['run_id']
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'latents').mkdir(); (directory / 'checkpoints').mkdir()
    started = time.monotonic()
    configure_strict_cuda_determinism(0)
    oracle = VAELatentOracle(vae=runtime['vae'], initial_latent=initial.to(runtime['vae_device']), compute_dtype=torch.float32)
    parameters = [p for p in oracle.parameters() if p.requires_grad]
    if len(parameters) != 1 or parameters[0] is not oracle.latent_fp32:
        raise RuntimeError('Only final z may be optimized')
    optimizer = torch.optim.Adam([oracle.latent_fp32], lr=.05, betas=(.9,.999), eps=1e-8,
                                 weight_decay=0., amsgrad=False)
    initial_hash = canonical_tensor_sha256(initial)
    if initial_hash != canonical_tensor_sha256(oracle.latent_fp32):
        raise RuntimeError('Initialization changed during device copy')
    replay.write_json(directory / 'manifest.json', {**spec, 'initial_latent_sha256': initial_hash,
        'reference_sha256': canonical_tensor_sha256(reference), 'optimizer_steps': 256,
        'fresh_start': True, 'training_prompts': ['original_open'], 'gold_eos_appended': True,
        'loss': 'mean_answer_ce + eos_ce', 'termination_contract': runtime['termination']})
    try:
        replay.write_json(directory / 'initial_reproducibility.json', gradient_gate(oracle, runtime))
        for step in range(257):
            current = oracle.latent_fp32.detach().cpu().clone()
            path = directory / 'latents' / f'step-{step:03d}.pt'
            sha = storage.save_tensor_payload(path, {'optimizer_step': step, 'latent_fp32': current})
            replay.append_jsonl(directory / 'latent_index.jsonl', {'optimizer_step': step,
                'path': path.relative_to(directory).as_posix(), 'file_sha256': sha,
                'latent_sha256': canonical_tensor_sha256(current)})
            if step in CHECKPOINTS:
                path = directory / 'checkpoints' / f'step-{step:03d}.pt'
                sha = storage.save_tensor_payload(path, {'optimizer_step': step, 'latent_fp32': current,
                    'initial_latent_fp32': initial, 'optimizer': optimizer.state_dict(), 'rng': storage.capture_rng()})
                replay.append_jsonl(directory / 'checkpoint_index.jsonl', {'optimizer_step': step,
                    'path': path.relative_to(directory).as_posix(), 'file_sha256': sha})
                with torch.no_grad():
                    pixels = oracle.image().cpu()
                evaluate(runtime, pixels, spec=spec, step=step, directory=directory, all_prompts=False)
                from PIL import Image
                rgb=(pixels[0].float().clamp(0,1)*255).round().byte().permute(1,2,0).numpy()
                Image.fromarray(rgb).save(directory / f'step-{step:03d}.png')
                del pixels
            if step == 256:
                break
            optimizer.zero_grad(set_to_none=True)
            output = teacher(runtime, oracle.image(), 'original_open', True)
            if not torch.isfinite(output.loss):
                raise RuntimeError('Nonfinite loss')
            output.loss.backward()
            gradient = oracle.latent_fp32.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError('Invalid latent gradient')
            loss, answer, eos = (float(x.detach()) for x in (output.loss, output.answer_loss, output.eos_loss))
            grad_rms = float(gradient.detach().float().square().mean().sqrt())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updated = oracle.latent_fp32.detach().cpu()
            if not torch.isfinite(updated).all():
                raise RuntimeError('Nonfinite updated latent')
            metrics = {**spec, 'optimizer_step': step+1, 'loss_before_step': loss,
                'answer_ce_before_step': answer, 'eos_ce_before_step': eos,
                'gradient_rms': grad_rms, 'update_rms': float((updated-current).square().mean().sqrt()),
                'latent_rms': float(updated.square().mean().sqrt()),
                'delta_from_start_rms': float((updated-initial).square().mean().sqrt()),
                'elapsed_seconds': time.monotonic()-started}
            replay.append_jsonl(directory / 'metrics.jsonl', metrics)
            if step % 16 == 0:
                print(json.dumps({'stage':'optimizer_step','id':spec['run_id'],'step':step+1,
                                  'answer_ce':answer,'eos_ce':eos}), flush=True)
            del output, gradient
        with torch.no_grad():
            image = oracle.image().cpu()
        endpoint_sha = storage.save_tensor_payload(directory / 'endpoint_raw.pt',
            {**spec, 'optimizer_step':256, 'latent_fp32':oracle.latent_fp32, 'image':image})
        generated = evaluate(runtime, image, spec=spec, step=256, directory=directory, all_prompts=True)
        replay.frozen_audit(runtime['vae'], runtime['reader'])
        result = {**spec, 'status':'completed','optimizer_steps':256,'latent_count':257,
            'checkpoint_count':len(CHECKPOINTS), 'generation_count':len(generated),
            'endpoint_file_sha256':endpoint_sha, 'endpoint_latent_sha256':canonical_tensor_sha256(oracle.latent_fp32),
            'elapsed_seconds':time.monotonic()-started}
        replay.write_json(directory / 'terminal.json', result)
        return result
    except BaseException as error:
        replay.write_json(directory / 'terminal.json', {**spec,'status':'failed','error':str(error)})
        raise
