"""Prepare a hashed tail to finish a timed-out artifact transfer without restarting it."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--offset', type=int, required=True)
p.add_argument('--output-prefix', type=Path, required=True)
a = p.parse_args()
size = a.source.stat().st_size
if not 0 < a.offset < size:
    raise ValueError('Offset must be inside the original sealed file')
tail_path = Path(str(a.output_prefix) + '.bin')
meta_path = Path(str(a.output_prefix) + '.json')
if tail_path.exists() or meta_path.exists():
    raise ValueError('Tail output already exists')
full_hash, prefix_hash, tail_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
remaining = a.offset
with a.source.open('rb') as source, tail_path.open('xb') as tail:
    while block := source.read(8 * 1024 * 1024):
        full_hash.update(block)
        prefix = block[:remaining]
        prefix_hash.update(prefix)
        remaining -= len(prefix)
        suffix = block[len(prefix):]
        tail_hash.update(suffix)
        tail.write(suffix)
meta = {'source': str(a.source), 'source_bytes': size, 'prefix_bytes': a.offset,
        'tail_bytes': size - a.offset, 'source_sha256': full_hash.hexdigest(),
        'prefix_sha256': prefix_hash.hexdigest(), 'tail_sha256': tail_hash.hexdigest()}
meta_path.write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps(meta))
