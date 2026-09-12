"""Export a completed full-U-Net endpoint without optimizer or teacher-bank payloads."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from vision_memory.dreamlite.writer_package import export_completed_writer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--parent-run', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
manifest = export_completed_writer(a.parent_run, a.output)
print(json.dumps({'package': str(a.output), 'status': manifest['status'], 'weights_sha256': manifest['weights_sha256']}))
