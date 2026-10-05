"""Select teacher checkpoints on disjoint training-history queries, using real PNGs."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import torch
from scripts.experiments.prefeval_context_coverage import VALIDATION, family_mean, choose_checkpoint
from scripts.experiments.prefeval_k1_data import load_training_records, sha
from scripts.experiments.prefeval_multitarget_bank import select_rows
from scripts.experiments.prefeval_prompt_matching import get_history_target, digest
from scripts.experiments.prefeval_k1_teacher import save_json
from scripts.eval.prefeval_rgb import load_reader, read_png
from vision_memory.reader.open_eos import assistant_termination_contract
from vision_memory.reader.prompt_matching import qwen3vl_continuation_logits, soft_target_kl_divergence
from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT
from vision_memory.repro import configure_strict_cuda_determinism


@torch.no_grad()
def main(args):
    configure_strict_cuda_determinism(0)
    torch.set_num_threads(1)
    rows = select_rows(load_training_records('pilot', False), args.ids_file)
    processor, reader = load_reader(args.reader, args.device)
    termination = assistant_termination_contract(reader, processor)
    gray = torch.full((3, 1024, 1024), 128/255, dtype=torch.float32, device=args.device)
    for row in rows:
        name = row['base_pair_id'].replace(':', '_')
        source, dest = args.teachers / name, args.output / name
        done = json.loads((source / 'complete.json').read_text())
        if done['step'] != 288 or done['binding']['context_suite'] != 'diverse-v1':
            raise ValueError('Selection requires a completed registered diverse teacher')
        expected = done['binding']['snapshot_steps']
        selection_binding = {'source_complete_sha256': sha(source / 'complete.json'),
                             'validation': VALIDATION, 'steps': expected,
                             'selector_sha256': sha(Path(__file__)), 'reader': done['binding']['reader_weights_sha256']}
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / 'complete.json').exists():
            old = json.loads((dest / 'complete.json').read_text())
            if old['selection_binding_sha256'] != digest(selection_binding) or old['png_sha256'] != sha(dest / 'memory.png'):
                raise ValueError('Selected endpoint identity changed')
            if old['selection_sha256'] != sha(dest / 'selection.json') or old['latent_sha256'] != sha(dest / 'latent.pt'):
                raise ValueError('Selected endpoint artifacts changed')
            continue
        targets, sensitivity = [], []
        for family, query in VALIDATION:
            item, path = get_history_target(cache_root=args.cache, binding=done['binding'], row=row,
                query=query, model=reader, processor=processor, reference=gray, device=args.device,
                assistant_end_token_id=termination['assistant_end_token_id'])
            targets.append((item, path, family, query))
            empty = qwen3vl_continuation_logits(model=reader, processor=processor, image=gray,
                query=query, target_ids=torch.tensor([item['target_ids']]), device=args.device,
                require_image_grad=False, reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
            sensitivity.append(float(soft_target_kl_divergence(empty.target_logits, item['logits'].to(args.device))))
        results = []
        for step in expected:
            snap = source / 'snapshots' / f'step-{step:04d}'
            receipt = json.loads((snap / 'snapshot.json').read_text())
            if (receipt['binding'] != done['binding'] or receipt['step'] != step
                    or receipt['png_sha256'] != sha(snap / 'memory.png')
                    or receipt['latent_sha256'] != sha(snap / 'latent.pt')):
                raise ValueError('Snapshot identity or bytes changed')
            pixels = read_png(snap / 'memory.png').to(args.device)
            scores = []
            for item, path, family, query in targets:
                output = qwen3vl_continuation_logits(model=reader, processor=processor, image=pixels,
                    query=query, target_ids=torch.tensor([item['target_ids']]), device=args.device,
                    require_image_grad=False, reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
                scores.append(float(soft_target_kl_divergence(output.target_logits, item['logits'].to(args.device))))
            results.append({'step': step, 'score': family_mean(scores), 'per_query_kl': scores,
                            'png_sha256': receipt['png_sha256'], 'latent_sha256': receipt['latent_sha256']})
        best = choose_checkpoint(results, expected)
        selected = source / 'snapshots' / f"step-{best['step']:04d}"
        selection = {'binding': selection_binding, 'candidates': results, 'selected': best,
                     'history_sensitivity_gray_kl': sensitivity,
                     'validation_targets': [{'query': q, 'family': f, 'cache_sha256': sha(p),
                                              'cache_path': str(p)} for _, p, f, q in targets]}
        save_json(dest / 'selection.json', selection)
        shutil.copyfile(selected / 'memory.png', dest / 'memory.png')
        shutil.copyfile(selected / 'latent.pt', dest / 'latent.pt')
        save_json(dest / 'complete.json', {'pair_id': row['base_pair_id'], 'step': best['step'],
            'training_budget': 288, 'selection_schema': 'dreamlite.context-selection.v1',
            'binding': done['binding'], 'selection_binding_sha256': digest(selection_binding),
            'selection_sha256': sha(dest / 'selection.json'), 'png_sha256': sha(dest / 'memory.png'),
            'latent_sha256': sha(dest / 'latent.pt')})
        print(json.dumps({'selected': row['base_pair_id'], 'step': best['step'], 'validation_kl': best['score']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('teachers', 'reader', 'cache', 'output', 'ids-file'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    main(p.parse_args())
