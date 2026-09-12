import argparse
import json
from pathlib import Path
import torch
from PIL import Image,ImageDraw

p=argparse.ArgumentParser()
p.add_argument('--run',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
identity=json.loads((a.run/'train/identity.json').read_text())
phases={phase:[json.loads(s) for s in (a.run/'train'/phase/'generations.jsonl').read_text().splitlines()]
        for phase in ('baseline','trained')}
groups=sorted({r['question_id'] for records in phases.values() for r in records})
canvas=Image.new('RGB',(identity['eval_seeds']*224,len(groups)*2*305+45),'white')
draw=ImageDraw.Draw(canvas)
scope = 'full U-Net' if identity.get('trainable_scope') == 'full_unet' else f"LoRA rank {identity['lora_rank']}"
draw.text((8,8),f"Official FM on {identity['model_variant'].title()} ({scope}) | same {identity['eval_seeds']} noise seeds | before / after {identity['steps']} optimizer updates",fill='black')
for phase_index,phase in enumerate(('baseline','trained')):
    root=a.run/'train'/phase
    answers={(r['question_id'],r['noise_seed']):r for r in phases[phase]
             if r['condition']=='matched' and r['prompt_id']=='original_open'}
    columns={qid:0 for qid in groups}
    for path in sorted(root.glob('*-seed-*.pt')):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        qid=payload['question_id']
        col=columns[qid]
        if col>=identity['eval_seeds']:
            raise ValueError('More images than the registered seeds for a condition')
        columns[qid]+=1
        record=answers[(qid,payload['noise_seed'])]
        pixels=(payload['image'][0].permute(1,2,0).float().clamp(0,1)*255).round().byte().numpy()
        image=Image.fromarray(pixels).resize((220,220))
        x,y=col*224,(groups.index(qid)*2+phase_index)*305+40
        canvas.paste(image,(x,y))
        draw.text((x+3,y+222),f"{phase} seed {col}",fill='black')
        draw.text((x+3,y+235),f"gold: {record['gold']}",fill='black')
        raw=record['raw']
        draw.text((x+3,y+248),raw[:31],fill='black')
        if len(raw)>31: draw.text((x+3,y+261),raw[31:62],fill='black')
        passed=record['scorer']['strict_correct'] and record['scorer']['answer_followed_immediately_by_eos']
        draw.text((x+3,y+274),'Original question: '+('PASS + EOS' if passed else 'FAIL'),fill='green' if passed else 'red')
    if any(count!=identity['eval_seeds'] for count in columns.values()):
        raise ValueError('Missing condition images')
a.output.parent.mkdir(parents=True,exist_ok=True)
canvas.save(a.output)
print(a.output)
