"""Show fixed first samples from confirmation or chained validation, without sample selection."""
import argparse
import json
from pathlib import Path
import textwrap
from PIL import Image, ImageDraw

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--kind', choices=('confirmation', 'chains'), required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
done = json.loads((a.run / 'complete.json').read_text())
rows = [json.loads(line) for line in (a.run / 'generations.jsonl').read_text().splitlines()]
original = [row for row in rows if row['prompt_id'] == 'original_open']
samples = []
if a.kind == 'confirmation':
    for style in ('trained', 'reworded_1', 'reworded_2'):
        for state in ('ambient', 'jazz', 'clear'):
            case = state + '_' + style
            plan = next(case_plan for case_plan in done['identity']['plan'] if case_plan['case'] == case)
            row = next(row for row in original if row['case'] == case and row['noise_seed'] == plan['seeds'][0])
            samples.append((case, row, Path(row['image_artifact']).with_suffix('.png')))
    cols, tile = 3, 332
    subtitle = 'First preregistered noise only per case; all 72 generated images / 360 answers are in JSON.'
else:
    for sequence in range(4):
        for step in range(6):
            row = next(row for row in original if row['sequence'] == sequence and row['repetition'] == 0 and row['step'] == step)
            label = f"order {sequence}, step {step}: {row['operation']}"
            samples.append((label, row, Path(row['image_artifact'])))
    cols, tile = 6, 204
    subtitle = 'First repetition of each preregistered order; all 16 chains / 96 writes / 480 answers are in JSON.'
height = tile + 91
canvas = Image.new('RGB', (cols * (tile + 8), ((len(samples) + cols - 1) // cols) * height + 40), 'white')
draw = ImageDraw.Draw(canvas)
draw.text((5, 3), 'Same full-U-Net checkpoint | native CFG1, image CFG1, 28 steps | fixed sample illustration', fill='black')
draw.text((5, 17), subtitle, fill='black')
for i, (label, row, filename) in enumerate(samples):
    x, y = (i % cols) * (tile + 8), (i // cols) * height + 40
    with Image.open(a.run / filename) as source:
        canvas.paste(source.resize((tile, tile)), (x, y))
    draw.text((x + 2, y + tile + 2), label, fill='black')
    draw.text((x + 2, y + tile + 15), 'Expected: ' + row['gold'], fill='black')
    for line, text in enumerate(textwrap.wrap('Raw: ' + row['raw'], width=tile // 6)[:3]):
        draw.text((x + 2, y + tile + 28 + line * 12), text, fill='black')
    passed = row['scorer']['strict_correct'] and row['scorer']['answer_followed_immediately_by_eos']
    draw.text((x + 2, y + tile + 68), 'Original query: ' + ('PASS + EOS' if passed else 'FAIL'), fill='green' if passed else 'red')
a.output.parent.mkdir(parents=True, exist_ok=True)
canvas.save(a.output)
print(a.output)
