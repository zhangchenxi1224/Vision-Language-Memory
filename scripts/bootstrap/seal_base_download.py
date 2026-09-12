"""Verify HF download content hashes and save a read-only base model binding."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
# This bootstrap also runs on an internet CPU box without PyTorch installed.
# Import the stdlib-only verifier without the repro package's CUDA helpers.
sys.path.insert(0,str(ROOT/"src/vision_memory/repro"))
from hf_snapshot import inspect_download


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--revision",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=inspect_download(a.model,a.revision)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists() and json.loads(a.output.read_text())!=result:
        raise RuntimeError("Existing model seal differs; do not overwrite")
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"files":len(result["files"]),"payload_sha256":result["payload_sha256"]}))
