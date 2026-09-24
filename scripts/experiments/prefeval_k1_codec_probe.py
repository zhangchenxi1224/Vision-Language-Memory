"""Fixed first-case, read-only native VAE roundtrip diagnostic (T1 MCQ)."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import torch
from PIL import Image
from diffusers import AutoencoderTiny
from scripts.experiments.prefeval_k1_data import load_records, official_mcq, option_order, sha
from scripts.experiments.prefeval_k1_teacher import save_json
from scripts.eval.prefeval_rgb import load_reader, read_png, append
from scripts.train.latent_r11_vae_oracle import _save_image
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
from vision_memory.reader.open_answer import generate_short_answer
from vision_memory.repro import configure_strict_cuda_determinism
from vision_memory.training.latent_bank_unet import OFFICIAL_REFERENCE_COMMIT


@torch.no_grad()
def main(args):
    configure_strict_cuda_determinism(0)
    torch.set_num_threads(1)
    assert not args.output.exists(), 'Inspect existing probe; never overwrite a run'
    official = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.official_source, text=True).strip()
    assert official == OFFICIAL_REFERENCE_COMMIT
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=args.official_source, text=True).strip()
    sys.path.insert(0, str(args.official_source))
    from dreamlite import DreamLitePipelineLoRA
    row = load_records('pilot')[0]
    assert row['base_pair_id'] == 'entertain_games:0001'
    pair = row['base_pair_id'].replace(':', '_')
    cases = []
    for arm in ['A', 'B']:
        teacher = args.run / 'pilot' / arm / pair
        done = json.loads((teacher / 'complete.json').read_text())
        assert done['step'] == 288
        cases.append(dict(arm=arm, stage='teacher', chain=0, path=teacher / 'memory.png', expected_sha=done['png_sha256']))
        for stage in ['write-pilot', 'final-pilot']:
            images = args.run / 'writer' / arm / stage
            manifest = json.loads((images / 'manifest.json').read_text())
            for chain in [0, 1]:
                folder = images / pair / f'seed-{chain}'
                done = json.loads((folder / 'complete.json').read_text())
                assert done['binding'] == manifest
                cases.append(dict(arm=arm, stage=stage, chain=chain, path=folder / 'prefix-00.png',
                    expected_sha=done['png_hashes']['prefix-00.png'], checkpoint_sha256=manifest['checkpoint_sha256']))
    for case in cases:
        assert sha(case['path']) == case['expected_sha']
    args.output.mkdir(parents=True)
    vae = AutoencoderTiny.from_pretrained(args.base, subfolder='vae', local_files_only=True,
        torch_dtype=torch.float32).to(args.device).eval().requires_grad_(False)
    assert vae.config.latent_channels == 4 and float(vae.config.scaling_factor) == 1 and float(vae.config.shift_factor) == 0
    # Invoke the real constructor and method, loading only the codec needed here.
    pipe = DreamLitePipelineLoRA(text_encoder=None, tokenizer=None, processor=None,
        vae=vae, unet=None, scheduler=None)
    manifest = dict(protocol='fixed_first_case_native_vae_roundtrip_v1', pair_id=row['base_pair_id'],
        official_commit=official, official_pipeline_sha256=sha(Path(inspect.getfile(DreamLitePipelineLoRA))),
        implementation_sha256=sha(Path(__file__)),
        execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        cases=[{**c, 'path':str(c['path'])} for c in cases], reader=str(args.reader), base=str(args.base),
        image_processor=dict(pipe.image_processor.config), vae_dtype='float32', family='T1', task='mcq',
        transformations=['original saved PNG', 'official prepare_image_latents -> decode -> saved/reopened uint8 PNG'],
        optimizer_steps=0, max_new_tokens=32)
    save_json(args.output / 'manifest.json', manifest)
    endpoints = []
    for c in cases:
        source = Image.open(c['path']).convert('RGB')
        assert source.size == (1024, 1024)
        latent = pipe.prepare_image_latents(pipe.image_processor.preprocess(source), dtype=torch.float32, device=args.device)
        assert tuple(latent.shape) == (1, 4, 128, 128) and torch.isfinite(latent).all()
        rgb = decode_model_latents_unit_interval(vae, latent)
        target = args.output / f"{c['arm']}-{c['stage']}-seed-{c['chain']}-roundtrip.png"
        _save_image(target, rgb)
        before, after = read_png(c['path']), read_png(target)
        metric = dict(arm=c['arm'], stage=c['stage'], chain=c['chain'], source_png_sha256=sha(c['path']),
            output_png_sha256=sha(target), pixel_mse=float((before-after).square().mean()),
            pixel_mae=float((before-after).abs().mean()), latent_finite=True)
        append(args.output / 'codec.jsonl', metric)
        for transform, path in [('original', c['path']), ('roundtrip', target)]:
            endpoints.append((c, transform, path))
    del pipe, vae
    torch.cuda.empty_cache()
    processor, reader = load_reader(args.reader, args.device)
    mcq = official_mcq(ROOT / 'third_party/prefeval_reference')
    step = int.from_bytes(hashlib.sha256(f"eval:{row['base_pair_id']}:T1".encode()).digest()[:4], 'big')
    order, correct = option_order(row['base_pair_id'], step)
    query = row['forms']['T1'] + mcq['get_mcq_question_format']([row['options'][i] for i in order])
    for c, transform, path in endpoints:
        generated = generate_short_answer(model=reader, processor=processor, image=read_png(path),
            query=query, device=args.device, max_new_tokens=32)
        predicted = mcq['extract_choice'](generated['raw'])
        record = dict(pair_id=row['base_pair_id'], arm=c['arm'], stage=c['stage'], chain=c['chain'],
            transform=transform, png_path=str(path), png_sha256=sha(path), reader_query=query,
            family='T1', task='mcq', option_order=order, correct_letter='ABCD'[correct],
            predicted_letter=predicted, correct=predicted=='ABCD'[correct], parse_failure=predicted is None,
            generated=generated, max_new_tokens=32)
        append(args.output / 'readback.jsonl', record)
        print(json.dumps({k:record[k] for k in ['arm','stage','chain','transform','predicted_letter','correct']}), flush=True)
    save_json(args.output / 'complete.json', dict(cases=len(cases), readbacks=len(endpoints),
        manifest_sha256=sha(args.output / 'manifest.json'), readback_sha256=sha(args.output / 'readback.jsonl')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'base', 'reader', 'official-source', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    main(p.parse_args())
