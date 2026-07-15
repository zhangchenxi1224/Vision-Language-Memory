# Cluster PyTorch lock

`runtime-pinned.txt` deliberately excludes PyTorch. The exact `torch` and `torchvision`
pair must be selected after inspecting the target cluster driver, CUDA runtime, and GPU
architecture. Record the installed `torch` version in `VLM_EXPECTED_TORCH` and require
`scripts/bootstrap/preflight.py --mode cluster` to pass before running a real probe.

Do not create a comments-only requirements file: pip would report success without
installing or pinning PyTorch.
