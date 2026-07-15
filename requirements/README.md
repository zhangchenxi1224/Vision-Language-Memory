# Cluster PyTorch lock

`runtime-pinned.txt` deliberately excludes PyTorch. The exact `torch` and `torchvision`
pair must be selected after inspecting the target cluster driver, CUDA runtime, and GPU
architecture. Record the installed `torch` version in `VLM_EXPECTED_TORCH` and require
`scripts/bootstrap/preflight.py --mode cluster` to pass before running a real probe.

Do not create a comments-only requirements file: pip would report success without
installing or pinning PyTorch.

The verified initial Fudan A800 profile is:

```text
Python 3.10
CUDA module 11.8
torch 2.4.1+cu118
torchvision 0.19.1+cu118
```

The executable installation procedure is kept in
`scripts/cluster/setup_fudan_a800.sh`.
