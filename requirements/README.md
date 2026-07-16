# Cluster PyTorch lock

`runtime-pinned.txt` deliberately excludes PyTorch and remains the generic/local dependency
set. A target runtime must select PyTorch only after inspecting its driver, CUDA runtime,
and GPU architecture.

R3 uses `inspire-ngc2502-pinned.txt`. It locks every application-level package explicitly,
including NumPy, Pillow, PyYAML, einops, and tqdm. The Inspire bootstrap installs this file
with `--no-deps` inside a `--system-site-packages` overlay venv, closes only missing
**non-Torch** transitive dependencies, and proves that the NGC image's Torch fingerprint was
not overlaid. `scripts/inspire/preflight_r3_h200.py` must pass with `--require-models` before
any scientific H200 stage.

Do not create a comments-only requirements file: pip would report success without
installing or pinning PyTorch.

The following Fudan A800 profile is historical R1/R2 evidence only:

```text
Python 3.10
CUDA module 11.8
torch 2.7.1+cu118
torchvision 0.22.1+cu118
```

The executable installation procedure is kept in
`scripts/cluster/setup_fudan_a800.sh`; it must not be used for R3.
