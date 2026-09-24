"""Fixed first-case one-step retention from training versus regenerated source PNGs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import torch
from PIL import Image
from scripts.experiments.prefeval_k1_writer import load_pipe, encode_source
from scripts.experiments.prefeval_k1_data import load_records, event_text, official_mcq, option_order, sha
from scripts.experiments.prefeval_k1_teacher import save_json
from scripts.eval.prefeval_rgb import load_reader, read_png, append
from scripts.train.latent_r11_vae_oracle import _save_image
from vision_memory.dreamlite.native_base import NativeBaseEditSampler
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
from vision_memory.training.latent_bank_unet import stable_seed
from vision_memory.reader.open_answer import generate_short_answer
from vision_memory.repro import configure_strict_cuda_determinism


@torch.no_grad()
def main(args):
    configure_strict_cuda_determinism(0)
    torch.set_num_threads(1)
    assert not args.output.exists(), 'Inspect prior probe instead of overwriting'
    rows = load_records('pilot')
    query_row = rows[0]
    assert query_row['base_pair_id'] == 'entertain_games:0001'
    donor = next(r for r in rows[1:] if r['topic'] == query_row['topic'])
    assert donor['base_pair_id'] == 'entertain_games:0007'
    images = args.retained / args.arm / 'pilot'
    train = args.retained / args.arm / 'train'
    args.checkpoint = train / 'checkpoint-final.pt'
    checkpoint_sha = sha(args.checkpoint)
    assert checkpoint_sha == json.loads((train/'complete.json').read_text())['checkpoint_sha256']
    assert checkpoint_sha == json.loads((images/'manifest.json').read_text())['checkpoint_sha256']
    source_hashes = json.loads((args.retained/'launch.json').read_text())['source_checks'][args.arm]['initial_source_hashes']
    cases = []
    for control, row in [('memory', query_row), ('mismatch', donor)]:
        pid = row['base_pair_id']
        for chain in [0, 1]:
            for source_kind in ['training', 'regenerated']:
                if source_kind == 'training':
                    folder = args.run/'writer'/args.arm/'training-prefixes'/pid.replace(':','_')/'seed-0'
                else:
                    folder = images/pid.replace(':','_')/f'seed-{chain}'
                source = folder/'prefix-00.png'
                expected = json.loads((folder/'complete.json').read_text())['png_hashes']['prefix-00.png']
                assert sha(source) == expected
                if source_kind == 'training':
                    assert expected == source_hashes[pid]
                cases.append(dict(control=control, source_pair_id=pid, chain=chain, source_kind=source_kind,
                    source_path=str(source), source_sha256=expected, event=event_text(row['history'][2:4]),
                    noise_seed=stable_seed(20260924, f'rollout:{pid}:{chain}', 1)))
    args.output.mkdir(parents=True)
    save_json(args.output/'manifest.json', dict(arm=args.arm, checkpoint_sha256=checkpoint_sha,
        implementation_sha256=sha(Path(__file__)), query_pair_id=query_row['base_pair_id'], cases=cases,
        scope='Fixed first pair, original native 28-step one-step inference; not a full-chain or full64 score'))
    pipe = load_pipe(args)
    pipe.unet.requires_grad_(False)
    endpoints = []
    for c in cases:
        image = Image.open(c['source_path']).convert('RGB')
        source = encode_source(pipe, image, args.device)
        generator = torch.Generator(device=args.device).manual_seed(c['noise_seed'])
        noise = torch.randn(source.shape, generator=generator, device=args.device, dtype=source.dtype)
        sampler = NativeBaseEditSampler(pipe, source_image=image, event_text=c['event'], guidance_scale=1.)
        generated = sampler(source_latents=source, noise_latents=noise, num_steps=28, return_trajectory=False)
        dest = args.output/f"{c['source_kind']}-{c['control']}-{c['chain']}-after.png"
        _save_image(dest, decode_model_latents_unit_interval(pipe.vae, generated.latents, clamp=True))
        repeated_sha = None
        if c['source_kind'] == 'regenerated':
            recorded = images/c['source_pair_id'].replace(':','_')/f"seed-{c['chain']}"/'prefix-01.png'
            repeated_sha = sha(recorded)
            assert sha(dest) == repeated_sha, 'Identical native inference must reproduce saved PNG exactly'
        append(args.output/'writes.jsonl', dict(**c, output_path=str(dest), output_sha256=sha(dest),
                                               prior_prefix1_sha256=repeated_sha))
        endpoints.extend([(c, 'before', Path(c['source_path'])), (c, 'after', dest)])
        del sampler, generated, source, noise
    del pipe
    torch.cuda.empty_cache()
    processor, reader = load_reader(args.reader, args.device)
    mcq = official_mcq(ROOT/'third_party/prefeval_reference')
    step = int.from_bytes(hashlib.sha256(f"eval:{query_row['base_pair_id']}:T1".encode()).digest()[:4], 'big')
    order, correct = option_order(query_row['base_pair_id'], step)
    query = query_row['forms']['T1'] + mcq['get_mcq_question_format']([query_row['options'][i] for i in order])
    for c, stage, path in endpoints:
        generated = generate_short_answer(model=reader, processor=processor, image=read_png(path),
                                           query=query, device=args.device, max_new_tokens=32)
        predicted = mcq['extract_choice'](generated['raw'])
        record = dict(pair_id=query_row['base_pair_id'], arm=args.arm, source_pair_id=c['source_pair_id'],
            control=c['control'], chain=c['chain'], source_kind=c['source_kind'], stage=stage,
            png_path=str(path), png_sha256=sha(path), family='T1', task='mcq', reader_query=query,
            option_order=order, correct_letter='ABCD'[correct], predicted_letter=predicted,
            correct=predicted=='ABCD'[correct], parse_failure=predicted is None, generated=generated)
        append(args.output/'readback.jsonl', record)
        print(json.dumps({k:record[k] for k in ['control','chain','source_kind','stage','predicted_letter','correct']}), flush=True)
    save_json(args.output/'complete.json', dict(cases=8, readbacks=16,
        readback_sha256=sha(args.output/'readback.jsonl'), manifest_sha256=sha(args.output/'manifest.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['A','B'], required=True)
    for name in ['run','retained','base','reader','official-source','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    main(p.parse_args())
