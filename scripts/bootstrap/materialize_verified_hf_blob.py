"""Reuse a cached blob only when it matches the destination revision's HF hash."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--relative",required=True)
    p.add_argument("--revision",required=True)
    p.add_argument("--sha256",required=True)
    p.add_argument("--audit",type=Path,required=True)
    a=p.parse_args()
    root=a.model.resolve()
    target=(root/a.relative).resolve()
    target.relative_to(root)
    if sha(a.source)!=a.sha256: raise ValueError("Cached blob is not the official target content")
    method="existing_verified_file"
    if target.exists():
        if sha(target)!=a.sha256: raise ValueError("Different destination file already exists")
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
        try:
            os.link(a.source,target)
            method="hardlink_from_identical_verified_blob"
        except OSError:
            with a.source.open("rb") as src,target.open("xb") as dst: shutil.copyfileobj(src,dst,8*1024*1024)
            method="copy_from_identical_verified_blob"
        if sha(target)!=a.sha256: raise ValueError("Materialized blob verification failed")
    metadata=root/".cache/huggingface/download"/(a.relative+".metadata")
    metadata.parent.mkdir(parents=True,exist_ok=True)
    if not metadata.exists():
        metadata.write_text(f"{a.revision}\n{a.sha256}\n{time.time()}\n")
    a.audit.parent.mkdir(parents=True,exist_ok=True)
    audit={"source":str(a.source),"destination":str(target),"target_revision":a.revision,"official_target_sha256":a.sha256,
        "method":method,"provenance":"HF base revision content hash verified against cached Mobile blob; no model substitution"}
    a.audit.write_text(json.dumps(audit,sort_keys=True,indent=2)+"\n")
    print(json.dumps(audit))
