"""Illustrate the first paired seed; all eight seeds remain in the result tables."""
import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import torch
from PIL import Image, ImageDraw

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--parent-run',type=Path,required=True)
p.add_argument('--native-control',type=Path,required=True)
p.add_argument('--raw-control',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
cases=[('native CFG7.5',a.parent_run/'train/trained'),('native CFG1',a.native_control/'guidance1'),
       ('training condition CFG1',a.raw_control/'training_raw_guidance1')]
canvas=Image.new('RGB',(1020,1235),'white')
draw=ImageDraw.Draw(canvas)
draw.text((10,8),'Same full-U-Net checkpoint | first paired seed illustration | all 8 seeds / 5 prompts are reported in JSON',fill='black')
for row,(label,directory) in enumerate(cases):
    records=[json.loads(s) for s in (directory/'generations.jsonl').read_text().splitlines()]
    for col,state in enumerate(('ambient','jazz','clear')):
        qids={r['question_id'] for r in records if r['question_id'].endswith('-state-'+state)}
        if len(qids)!=1:raise ValueError('Expected exactly one group per state')
        qid=qids.pop()
        filename=hashlib.sha256(qid.encode()).hexdigest()[:16]+'-seed-00.pt'
        payload=torch.load(directory/filename,map_location='cpu',weights_only=True)
        record=next(r for r in records if r['question_id']==qid and r['condition']=='matched'
                    and r['prompt_id']=='original_open' and r['noise_seed']==payload['noise_seed'])
        pixels=(payload['image'][0].permute(1,2,0).clamp(0,1)*255).round().byte().numpy()
        x,y=col*340,row*400+35
        canvas.paste(Image.fromarray(pixels).resize((332,332)),(x,y))
        draw.text((x+3,y+334),label+' | '+record['gold'],fill='black')
        for line,text in enumerate(textwrap.wrap(record['raw'],width=52)[:2]):
            draw.text((x+3,y+348+line*12),text,fill='black')
        passed=record['scorer']['strict_correct'] and record['scorer']['answer_followed_immediately_by_eos']
        draw.text((x+3,y+375),'Original question: '+('PASS + EOS' if passed else 'FAIL'),fill='green' if passed else 'red')
a.output.parent.mkdir(parents=True,exist_ok=True)
canvas.save(a.output)
print(a.output)
