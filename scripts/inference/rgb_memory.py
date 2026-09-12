"""Run event-only writes and answer-blind reads with an exported official RGB Writer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def validate_command(command):
    if not isinstance(command, dict):
        raise ValueError('Each command must be a JSON object')
    if command.get('op') == 'write' and set(command) == {'op', 'event', 'seed'}:
        if (not isinstance(command['event'], str) or not command['event'].strip()
                or type(command['seed']) is not int or not 0 <= command['seed'] < 2**63):
            raise ValueError('A write requires nonempty event text and an explicit63-bit seed')
    elif command.get('op') == 'read' and set(command) == {'op', 'query'}:
        if not isinstance(command['query'], str) or not command['query'].strip():
            raise ValueError('A read requires a nonempty question')
    else:
        raise ValueError('Only write(event, seed) and read(query) commands are supported')
    return command


def run_commands(memory, commands, read_image, output):
    from vision_memory.dreamlite.writer_package import file_sha
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, writes = [], 0
    for index, command in enumerate(commands):
        validate_command(command)
        if command['op'] == 'write':
            result = memory.write(command['event'], seed=command['seed'])
            writes += 1
            path = output / f'memory-{writes:04d}.png'
            result.image.save(path)
            row = {'index': index, 'op': 'write', 'memory_image': path.name, 'image_file_sha256': file_sha(path)}
        else:
            before = memory.image.tobytes()
            answer = read_image(memory.image, command['query'])
            if memory.image.tobytes() != before:
                raise RuntimeError('Reader changed persistent memory')
            row = {'index': index, 'op': 'read', 'query': command['query'], **answer}
        rows.append(row)
        with (output / 'results.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    memory.image.save(output / 'memory-final.png')
    return {'commands': len(rows), 'writes': writes, 'reads': len(rows) - writes,
            'final_image_sha256': file_sha(output / 'memory-final.png')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--base-model', type=Path, required=True)
    p.add_argument('--official-source', type=Path, required=True)
    p.add_argument('--reader-model', type=Path, help='Required only when a read command occurs')
    p.add_argument('--commands', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--initial-image', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    from vision_memory.repro.determinism import REQUIRED_DETERMINISM_ENV
    import os
    if not a.worker:
        import subprocess
        return subprocess.call([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
                               env={**os.environ, **REQUIRED_DETERMINISM_ENV})
    for name, value in REQUIRED_DETERMINISM_ENV.items():
        if os.environ.get(name, value) != value:
            raise ValueError('Determinism environment differs: ' + name)
        os.environ[name] = value
    import numpy as np
    from PIL import Image
    import torch
    from vision_memory.repro import configure_strict_cuda_determinism
    from vision_memory.dreamlite.writer_package import load_writer_package, file_sha
    from vision_memory.dreamlite.rgb_memory import OfficialRGBMemory
    from vision_memory.reader.open_answer import generate_short_answer
    configure_strict_cuda_determinism(20260913)
    command_sha = file_sha(a.commands)
    package_sha = file_sha(a.package / 'manifest.json')
    commands = [validate_command(json.loads(line)) for line in a.commands.read_text(encoding='utf-8').splitlines() if line.strip()]
    if any(c['op'] == 'read' for c in commands) and a.reader_model is None:
        raise ValueError('Read commands require the frozen Reader snapshot')
    pipe, manifest = load_writer_package(a.package, base_model=a.base_model, official_source=a.official_source, device=a.device)
    initial = None
    if a.initial_image:
        with Image.open(a.initial_image) as image:
            initial = image.copy()
    memory = OfficialRGBMemory(pipe, image=initial, guidance_scale=manifest['guidance_scale'])
    loaded_reader = None

    def read_image(image, query):
        nonlocal loaded_reader
        if loaded_reader is None:
            from scripts.inspire.model_snapshot_manifest import verify_snapshot_manifest
            from scripts.train.r11_new_frozen_dreamlite_oracle import _load_reader
            from types import SimpleNamespace
            expected = manifest['reader_snapshot']
            actual = verify_snapshot_manifest(manifest_path=a.reader_model / '.snapshot_manifest.json', model_dir=a.reader_model,
                expected_repo_id=expected['repo_id'], expected_revision=expected['revision'])
            if any(actual[key] != expected[key] for key in ('manifest_sha256', 'snapshot_payload_sha256')):
                raise ValueError('Reader snapshot differs from the trained experiment')
            loaded_reader = _load_reader(SimpleNamespace(reader=a.reader_model), torch.device(a.device), torch.bfloat16)
        processor, reader = loaded_reader
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.
        return generate_short_answer(model=reader, processor=processor, image=pixels.to(a.device), query=query,
                                     device=a.device, max_new_tokens=32, do_sample=False)

    summary = run_commands(memory, commands, read_image, a.output)
    if file_sha(a.commands) != command_sha or file_sha(a.package / 'manifest.json') != package_sha:
        raise RuntimeError('Command file or package identity changed during inference')
    summary.update(package_manifest_sha256=package_sha,
                   package_status=manifest['status'], command_file_sha256=command_sha)
    (a.output / 'complete.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
