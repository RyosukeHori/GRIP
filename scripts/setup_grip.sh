#!/bin/bash
# One-shot installer for the GRIP Python deps.
# Assumes the conda env `grip` is already active and Isaac Gym has been installed.
#
# Three known gotchas are handled here:
#   1. PyTorch must use the CUDA 12.1 wheel (cu121) so that the numerical
#      kernels match the reference `isaac` env. The `pytorch-cuda=11.8`
#      conda build behaves differently on driver 590+, producing INT_MAX
#      values out of motion_lib's frame-count calculation and inflating
#      termination counts.
#   2. The SMPL forks must be installed before requirements.txt, otherwise
#      pip's resolver thrashes on hydra-core / gymnasium constraints.
#   3. chumpy 0.70 is incompatible with numpy >= 1.24 because it imports
#      ``np.bool`` / ``np.int`` / ``np.str``. A one-line patch fixes it.
#   4. rl_games 1.1.4 ships a `safe_load` that deserializes a checkpoint
#      directly onto the device it was saved on. That changes downstream
#      CUDA allocation order vs. the reference run and makes motion_lib's
#      frame-count overflow to INT_MAX on cuda:0. Force `map_location='cpu'`
#      so the load path matches the reference env.

set -euo pipefail

cd "$(dirname "$0")/.."  # repo root

echo "=== 1/5  Installing PyTorch 2.1.1 + CUDA 12.1 wheel ==="
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1

echo ""
echo "=== 2/5  Pinning numpy to the OpenBLAS pip wheel ==="
# conda may have left a netlib-BLAS numpy 1.22 behind; force the pip wheel.
pip install --force-reinstall --no-deps numpy==1.24.4

echo ""
echo "=== 3/5  Installing custom SMPL forks (anchors hydra/gymnasium versions) ==="
pip install "git+https://github.com/ZhengyiLuo/smplx.git@master"
pip install "git+https://github.com/ZhengyiLuo/SMPLSim.git@master"

echo ""
echo "=== 4/5  Installing the rest of requirements.txt ==="
pip install -r requirements.txt

echo ""
echo "=== 5/6  Patching chumpy (fixes ImportError on numpy >= 1.24) ==="
python - <<'PY'
import os
import chumpy

init_path = os.path.join(os.path.dirname(chumpy.__file__), '__init__.py')
old = "from numpy import bool, int, float, complex, object, unicode, str, nan, inf"
new = (
    "import numpy as _np\n"
    "bool = bool; int = int; float = float; complex = complex; "
    "object = object; unicode = str; str = str; "
    "nan = _np.nan; inf = _np.inf"
)
with open(init_path) as f:
    text = f.read()
if old in text:
    with open(init_path, 'w') as f:
        f.write(text.replace(old, new))
    print(f"  patched {init_path}")
else:
    print("  chumpy already patched (or unexpected layout) — nothing to do")
PY

echo ""
echo "=== 6/6  Patching rl_games safe_load (forces map_location=cpu) ==="
python - <<'PY'
import os
import rl_games.algos_torch.torch_ext as _te

path = _te.__file__
old = "    return safe_filesystem_op(torch.load, filename)\n"
new = "    return safe_filesystem_op(torch.load, filename, map_location=torch.device('cpu'))\n"
with open(path) as f:
    text = f.read()
if old in text:
    with open(path, 'w') as f:
        f.write(text.replace(old, new))
    print(f"  patched {path}")
elif "map_location=torch.device('cpu')" in text:
    print("  rl_games already patched — nothing to do")
else:
    print("  WARNING: rl_games safe_load signature changed — manual review needed")
PY

echo ""
echo "=== setup complete ==="
python -c "import torch; print(f'  torch={torch.__version__}  cuda={torch.version.cuda}  cudnn={torch.backends.cudnn.version()}')"
echo "Verify the full pipeline with:  python dynamics_net/run_hydra.py --help"
