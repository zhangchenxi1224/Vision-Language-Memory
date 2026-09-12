"""Render a scientific montage from verified DreamLite-generated PNGs."""
import argparse
import hashlib
import json
from pathlib import Path
import textwrap
from PIL import Image, ImageDraw

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--run",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
complete=json.loads((a.run/"complete.json").read_text())
for name,digest in complete["artifact_hashes"].items():
    if name.endswith((".json",".jsonl",".png")):
        if hashlib.sha256((a.run/name).read_bytes()).hexdigest()!=digest:
            raise ValueError("Source result or PNG changed")
records=[json.loads(x) for x in (a.run/"generations.jsonl").read_text().splitlines()]
if len(records)!=80:
    raise ValueError("Full fixed evaluation required")
seeds=complete["identity"]["noise_seeds"]
index={(r["event_index"],r["form"],r["noise_seed"]):r for r in records if r["prompt_id"]=="original_open"}
if len(index)!=16:
    raise ValueError("Incomplete original-question panel")
canvas=Image.new("RGB",(4*256,4*330+65),"white")
draw=ImageDraw.Draw(canvas)
draw.text((8,8),"Official Base, zero updates | Native 28 steps, CFG 7.5 | Same four fresh noises",fill="black")
draw.text((8,25),"Captions: raw original-question answers (long captions shortened only for preview).",fill="black")
for row,(event,form) in enumerate([(e,f) for e in (0,1) for f in ("raw_event","memory_note")]):
    for col,seed in enumerate(seeds):
        r=index[event,form,seed]
        name=r["image_artifact"].replace(".pt",".png")
        picture=Image.open(a.run/name).convert("RGB").resize((248,248))
        x,y=col*256,row*330+60
        canvas.paste(picture,(x,y))
        passed=r["scorer"]["strict_correct"] and r["scorer"]["answer_followed_immediately_by_eos"]
        draw.rectangle((x,y+250,x+248,y+267),fill="#dbf0de" if passed else "#f7e1df")
        draw.text((x+3,y+253),f"{r['gold']} | {form} | seed {col}",fill="black")
        lines=textwrap.wrap(r["raw"],width=39)
        for k,line in enumerate(lines[:3]):
            draw.text((x+3,y+273+14*k),line,fill="black")
a.output.parent.mkdir(parents=True,exist_ok=True)
canvas.save(a.output)
print(a.output)
