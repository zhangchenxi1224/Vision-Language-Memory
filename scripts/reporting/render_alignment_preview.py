import argparse
import json
from pathlib import Path
import torch
from PIL import Image,ImageDraw

p=argparse.ArgumentParser()
p.add_argument('--run',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
canvas=Image.new('RGB',(8*224,2*270+45),'white')
draw=ImageDraw.Draw(canvas)
identity=json.loads((a.run/'train/identity.json').read_text())
scope = 'full U-Net' if identity.get('trainable_scope') == 'full_unet' else f"LoRA rank {identity['lora_rank']}"
draw.text((8,8),f"Official FM on {identity['model_variant'].title()} ({scope}) | same {identity['eval_seeds']} noise seeds | before / after {identity['steps']} optimizer updates",fill='black')
for row,phase in enumerate(('baseline','trained')):
    root=a.run/'train'/phase
    records=[json.loads(s) for s in (root/'generations.jsonl').read_text().splitlines()]
    answers={r['noise_seed']:r['raw'] for r in records if r['condition']=='matched' and r['prompt_id']=='original_open'}
    for col,path in enumerate(sorted(root.glob('*-seed-*.pt'))):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        pixels=(payload['image'][0].permute(1,2,0).float().clamp(0,1)*255).round().byte().numpy()
        image=Image.fromarray(pixels).resize((220,220))
        x,y=col*224,row*270+40
        canvas.paste(image,(x,y))
        draw.text((x+3,y+222),f'{phase} seed {col}',fill='black')
        raw=answers[payload['noise_seed']]
        draw.text((x+3,y+236),raw[:31],fill='black')
        if len(raw)>31: draw.text((x+3,y+249),raw[31:62],fill='black')
a.output.parent.mkdir(parents=True,exist_ok=True)
canvas.save(a.output)
print(a.output)
