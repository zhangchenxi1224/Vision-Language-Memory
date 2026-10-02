"""Seal a CPU-audited successful-latent set only after a fresh EOS campaign ends."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from vision_memory.training.latent_teacher_bank import build_bank, write_json


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route",choices=("direct","frozen"),required=True)
    for name in ("oracle-root","oracle-repo","oracle-config","output-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--oracle-commit",required=True)
    p.add_argument("--planned-manifest",type=Path)
    p.add_argument("--progress-only",action="store_true")
    return p


def main():
    args=parser().parse_args()
    try:
        result=build_bank(**vars(args))
        print(json.dumps({key:result.get(key) for key in ("route","status","state","planned_count","audited_count","unique_teacher_count")}),flush=True)
        if result.get("status") == "blocked_no_success":
            return 10
        if result.get("status") != "sealed":
            return 11
        return 0
    except BaseException as exc:
        write_json(args.output_dir/"audit_failure.json",{"status":"failed","error":str(exc),"unet_training_started":False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
