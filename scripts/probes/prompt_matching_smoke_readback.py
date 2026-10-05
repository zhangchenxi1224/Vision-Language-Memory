"""Technical PNG readback for nonformal 12-step targets; never a benchmark score."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import torch
from scripts.experiments.prefeval_k1_data import load_training_records, official_mcq, sha
from scripts.experiments.prefeval_k1_teacher import query_target, save_json
from scripts.experiments.prefeval_prompt_matching import SUPERVISIONS
from scripts.eval.prefeval_rgb import load_reader, read_png
from vision_memory.reader.open_answer import generate_short_answer
from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT
from vision_memory.repro import configure_strict_cuda_determinism


@torch.no_grad()
def main(args):
    configure_strict_cuda_determinism(0)
    row = load_training_records('pilot')[0]
    mcq = official_mcq(ROOT / 'third_party/prefeval_reference')
    query, _, _ = query_target(row, args.arm, 0, mcq)
    processor, reader = load_reader(args.reader, args.device)
    result = {'scope': 'technical_smoke_only_not_benchmark_evidence',
              'pair_id': row['base_pair_id'], 'arm': args.arm, 'modes': {}}
    for mode in SUPERVISIONS:
        path = args.run / mode / 'teachers' / row['base_pair_id'].replace(':', '_')
        done = json.loads((path / 'complete.json').read_text())
        assert done['step'] == 12 and done['binding'].get('supervision', 'hard_ce') == mode
        assert done['png_sha256'] == sha(path / 'memory.png')
        records = [json.loads(line) for line in (path / 'optimization.jsonl').read_text().splitlines()]
        assert len(records) >= 12 and all(r['grad_norm'] > 0 for r in records)
        pixels = read_png(path / 'memory.png').to(args.device)
        generated = generate_short_answer(model=reader, processor=processor, image=pixels,
                         query=query, device=args.device, max_new_tokens=32 if args.arm == 'B' else 300,
                         reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
        result['modes'][mode] = {'png_sha256': done['png_sha256'], 'generated': generated,
                                'first_objective': records[0].get('objective_loss', records[0]['ce']),
                                'last_objective': records[-1].get('objective_loss', records[-1]['ce']),
                                'nonzero_gradients': True}
    save_json(args.run / 'technical-readback.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--reader', type=Path, required=True)
    p.add_argument('--arm', choices=['A', 'B'], required=True)
    p.add_argument('--device', default='cuda:0')
    main(p.parse_args())
