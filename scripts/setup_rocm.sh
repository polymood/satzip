#!/usr/bin/env sh
# Create .venv-rocm: satzip with the PyTorch build for AMD GPUs (ROCm).
# A separate environment, so that `uv run` and `uv sync` never replace it with the CPU build.
#
#   scripts/setup_rocm.sh                    # ROCm 10.0 (default)
#   ROCM=7.14 scripts/setup_rocm.sh          # ROCm 7.14, if the driver or GPU needs it
#
# Then:  .venv-rocm/bin/marimo edit notebooks/07_big_models.py
set -eu
cd "$(dirname "$0")/.."
ROCM="${ROCM:-10.0}"
TORCH="${TORCH:-2.15.0}"

uv venv --python 3.12 .venv-rocm
# --no-sources: ignore the CPU index that pyproject.toml sets for torch.
# The ROCm index comes first: torch and the rocm-sdk packages come from it, the rest from PyPI.
uv pip install --python .venv-rocm --no-sources \
    --index "https://download.pytorch.org/whl/rocm${ROCM}" --index-strategy unsafe-best-match \
    "torch==${TORCH}+rocm${ROCM}" -e . pytest

.venv-rocm/bin/python - <<'EOF'
import torch
print("torch", torch.__version__, "HIP", torch.version.hip)
if not torch.cuda.is_available():
    raise SystemExit("PyTorch does not see a GPU. Check the AMD driver and ROCm support for this GPU.")
p = torch.cuda.get_device_properties(0)
print("GPU:", p.name, f"{p.total_memory / 2**30:.1f} GB")
EOF
