#!/bin/bash
# ============================================================
# SeqTrust_Bench environment one-click setup script
# Goal: replicate the ChromDNA environment
# (Python 3.12.13 + torch 2.10.0+cu128)
#
# Usage:
#   bash setup_env.sh /path/to/flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
#
# The flash-attn wheel is NOT bundled with this repository
# (too large for GitHub). Obtain it separately and pass its
# local path as the first argument.
#
# If you have already created and activated a Python 3.12.13
# environment manually, you can simply run:
#   pip install -r requirements.txt
#   pip install /path/to/flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
# ============================================================
set -e

ENV_NAME="SeqTrust_Bench"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLASH_ATTN_WHL="$1"

# 0. Check that the flash-attn wheel path was provided and exists
if [ -z "${FLASH_ATTN_WHL}" ]; then
    echo "Usage: bash setup_env.sh /path/to/flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    echo "Note: the flash-attn wheel is not bundled with this repository; please obtain it separately."
    exit 1
fi
if [ ! -f "${FLASH_ATTN_WHL}" ]; then
    echo "Error: flash-attn wheel not found: ${FLASH_ATTN_WHL}"
    exit 1
fi

# 1. Create the conda environment (skipped if it already exists)
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo ">>> Environment ${ENV_NAME} already exists, skipping creation"
else
    echo ">>> Creating conda environment ${ENV_NAME} (Python 3.12.13)"
    conda create -n "${ENV_NAME}" python=3.12.13 -y
fi

# 2. Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# 3. Pin the pip version (same as the original environment)
echo ">>> Installing pip==26.1.1"
pip install --upgrade pip==26.1.1

# 4. Install all pip dependencies
#    (torch 2.10.0 automatically pulls in the cu128 nvidia libraries)
echo ">>> Installing requirements.txt (195 packages)"
pip install -r "${SCRIPT_DIR}/requirements.txt"

# 5. Install flash-attn from the local wheel
#    (build strictly matches torch 2.10 + cu12)
echo ">>> Installing flash-attn 2.8.1 from local wheel"
pip install "${FLASH_ATTN_WHL}"

# 6. Verify the installation
echo ">>> Verifying installation"
python -c "import torch; import flash_attn; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| flash-attn:', flash_attn.__version__)"

echo ">>> Done! Activate with: conda activate ${ENV_NAME}"
