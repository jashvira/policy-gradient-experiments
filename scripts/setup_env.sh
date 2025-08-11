#!/usr/bin/env bash

# Minimal environment setup for this repo (uv-first):
# - Uses `uv` (fast package manager) for venv + installs when available
# - Falls back to python venv + pip if uv is unavailable
# - Installs dependencies (torch, transformers, vllm, etc.) from pyproject
# - Optional: install flash-attn, login to HuggingFace, pre-download a model
#
# Usage examples:
#   bash scripts/setup_env.sh
#   PYTHON_BIN=python3.11 VENV_DIR=.venv bash scripts/setup_env.sh --flash-attn
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash scripts/setup_env.sh
#   bash scripts/setup_env.sh --hf-token $HF_TOKEN --download-model Qwen/Qwen2.5-Math-1.5B-Instruct --dest /mnt/models/qwen
#   bash scripts/setup_env.sh --download-qwen

set -Eeuo pipefail

section() { echo -e "\n\033[1;36m==> $*\033[0m"; }
info()    { echo -e "\033[0;32m[info]\033[0m $*"; }
warn()    { echo -e "\033[0;33m[warn]\033[0m $*"; }
err()     { echo -e "\033[0;31m[error]\033[0m $*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

# Configurable via env vars
# Let uv manage Python by default; we only need VENV_DIR and basic toggles
PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
USE_UV="${USE_UV:-1}"

# Flags / optional args
INSTALL_FLASH_ATTN=0
HF_TOKEN_ARG=""
DL_MODEL_ID=""
DL_DEST=""
CLEAR_VENV=0
RUN_SMOKE=0

# Convenience defaults for Qwen model used in this repo
QWEN_MODEL_ID_DEFAULT="Qwen/Qwen2.5-Math-1.5B-Instruct"
QWEN_DEST_DEFAULT="${REPO_ROOT}/.models/Qwen2.5-Math-1.5B-Instruct"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flash-attn)
      INSTALL_FLASH_ATTN=1
      shift
      ;;
    --clear-venv)
      CLEAR_VENV=1
      shift
      ;;
    --smoke)
      RUN_SMOKE=1
      shift
      ;;
    --download-qwen)
      DL_MODEL_ID="${DL_MODEL_ID:-$QWEN_MODEL_ID_DEFAULT}"
      DL_DEST="${DL_DEST:-$QWEN_DEST_DEFAULT}"
      shift
      ;;
    --hf-token)
      HF_TOKEN_ARG="$2"
      shift 2
      ;;
    --download-model)
      DL_MODEL_ID="$2"
      shift 2
      ;;
    --dest)
      DL_DEST="$2"
      shift 2
      ;;
    *)
      warn "Unknown arg: $1"; shift ;;
  esac
done

section "Environment"
DISPLAY_PY="${PYTHON_BIN}"
if [[ "${USE_UV}" -eq 1 ]]; then
  DISPLAY_PY="managed-by-uv"
fi
info "REPO_ROOT:   ${REPO_ROOT}"
info "PYTHON_BIN:  ${DISPLAY_PY}"
info "VENV_DIR:    ${VENV_DIR}"
info "TORCH_INDEX: ${TORCH_INDEX_URL:-<default>}"
info "USE_UV:      ${USE_UV}"

## No Python detection here when using uv; keep it simple per user preference

if [[ "${USE_UV}" -eq 1 ]]; then
  section "Ensuring uv is installed"
  # Make sure ~/.local/bin is on PATH in case uv installs there
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh || {
      warn "uv install failed; falling back to python venv + pip"
      USE_UV=0
    }
  fi
fi

section "Creating venv"
if [[ "${USE_UV}" -eq 1 ]]; then
  section "Ensuring uv is installed"
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
fi

if [[ "${USE_UV}" -eq 1 ]] && command -v uv >/dev/null 2>&1; then
  # Let uv fetch/manage an appropriate Python by default
  if [[ "${CLEAR_VENV}" -eq 1 ]]; then
    UV_VENV_CLEAR=1 uv venv --clear --python "${PYTHON_BIN:-3.11}" "${VENV_DIR}"
  else
    uv venv --python "${PYTHON_BIN:-3.11}" "${VENV_DIR}"
  fi
  info "Created (or reused) uv venv at ${VENV_DIR}"
else
  if [[ ! -d "${VENV_DIR}" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    info "Created venv at ${VENV_DIR}"
  else
    if [[ "${CLEAR_VENV}" -eq 1 ]]; then
      info "Clearing existing venv at ${VENV_DIR}"
      rm -rf "${VENV_DIR}"
      "${PYTHON_BIN}" -m venv "${VENV_DIR}"
      info "Recreated venv at ${VENV_DIR}"
    else
      info "Using existing venv at ${VENV_DIR}"
    fi
  fi
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
if [[ "${USE_UV}" -eq 0 ]]; then
  python -m pip install -U pip setuptools wheel
fi

section "Installing core dependencies"
if [[ "${USE_UV}" -eq 1 ]] && command -v uv >/dev/null 2>&1; then
  if [[ -n "${TORCH_INDEX_URL}" ]]; then
    info "Installing torch via uv from ${TORCH_INDEX_URL}"
    uv pip install --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio || {
      warn "Torch install via index failed; falling back to default"
      uv pip install torch torchvision torchaudio
    }
  fi
  # Install dependencies only (avoid editable install of this repo)
  uv pip install \
    datasets>=4.0.0 \
    evaluate>=0.4.5 \
    "torch>=2.6" \
    "transformers>=4.51.0" \
    "trl<0.20" \
    "peft<0.16" \
    "vllm>=0.10.0" \
    "wandb>=0.21.0" \
    "matplotlib>=3.10.3" \
    latex2sympy2-extended \
    math-verify \
    pylatexenc \
    sympy \
    "PyYAML>=6.0.2" \
    "pytest>=8.4.1" \
    "huggingface_hub>=0.23"
else
  if [[ -n "${TORCH_INDEX_URL}" ]]; then
    info "Installing torch from ${TORCH_INDEX_URL}"
    python -m pip install --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio || {
      warn "Torch install via index failed; falling back to default PyPI."
      python -m pip install torch torchvision torchaudio
    }
  fi
  # Install dependencies only (avoid editable install of this repo)
  python -m pip install \
    datasets>=4.0.0 \
    evaluate>=0.4.5 \
    "torch>=2.6" \
    "transformers>=4.51.0" \
    "trl<0.20" \
    "peft<0.16" \
    "vllm>=0.10.0" \
    "wandb>=0.21.0" \
    "matplotlib>=3.10.3" \
    latex2sympy2-extended \
    math-verify \
    pylatexenc \
    sympy \
    "PyYAML>=6.0.2" \
    "pytest>=8.4.1" \
    "huggingface_hub>=0.23"
fi

if [[ "${INSTALL_FLASH_ATTN}" -eq 1 ]]; then
  section "Installing flash-attn (optional)"

  # Determine which Python to use for testing
  PYTHON_CMD="python"
  if [[ "${USE_UV}" -eq 1 ]] && command -v uv >/dev/null 2>&1; then
    # Use uv's Python for consistent environment
    PYTHON_CMD="uv run python"
  fi

  # 1) Try prebuilt wheel (fast path, no toolkit required)
  info "Attempting prebuilt wheel installation..."
  if [[ "${USE_UV}" -eq 1 ]] && command -v uv >/dev/null 2>&1; then
    uv pip install --no-build-isolation --only-binary=:all: flash-attn || true
  else
    python -m pip install --no-build-isolation --only-binary=:all: flash-attn || true
  fi

  # Test if flash-attn is working
  if ${PYTHON_CMD} -c 'import flash_attn; print("OK")' >/dev/null 2>&1; then
    info "flash-attn installed via prebuilt wheel."
  else
    # 2) Ensure torch is available for build requirements (critical for uv)
    info "Prebuilt wheel failed, ensuring torch is available for build..."
    if [[ "${USE_UV}" -eq 1 ]] && command -v uv >/dev/null 2>&1; then
      # Make sure torch is installed in the uv environment
      uv pip install "torch>=2.0" || warn "Could not ensure torch for build"
    fi

    # 3) Prepare for source build
    warn "flash-attn wheel not available; attempting source build (this can be slow)."

    # Compute architectures to build for
    export TORCH_CUDA_ARCH_LIST="$(${PYTHON_CMD} - <<'PY'
import torch
arches=set()
try:
    n=torch.cuda.device_count()
    for i in range(n):
        major, minor = torch.cuda.get_device_capability(i)
        arches.add(f"{major}.{minor}")
except Exception:
    pass
print(";".join(sorted(arches)) or "8.0;8.6;8.9;9.0")
PY
)"
    export USE_NINJA=1
    export MAX_JOBS=${MAX_JOBS:-4}

    # Show CUDA toolchain hints
    if [[ -z "${CUDA_HOME:-}" ]]; then
      warn "CUDA_HOME not set; for source builds you usually need a matching CUDA toolkit (e.g., /usr/local/cuda-12.1)."
      warn "Set CUDA_HOME, PATH, and LD_LIBRARY_PATH appropriately if the build fails."
    else
      info "Using CUDA_HOME=${CUDA_HOME}"
      export PATH="$CUDA_HOME/bin:$PATH"
      export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    fi

    # Ensure build prerequisites
    if [[ "${USE_UV}" -eq 1 ]] && command -v uv >/dev/null 2>&1; then
      uv pip install ninja packaging || true
      # Critical: use --no-build-isolation with torch already available
      uv pip install --no-build-isolation flash-attn || true
    else
      python -m pip install ninja packaging || true
      python -m pip install --no-build-isolation flash-attn || true
    fi

    # Final test
    if ${PYTHON_CMD} -c 'import flash_attn; print("OK")' >/dev/null 2>&1; then
      info "flash-attn installed from source."
    else
      warn "flash-attn installation failed; continuing without."
    fi
  fi

  # Show flash-attn status
  if ${PYTHON_CMD} -c 'import flash_attn; print(f"flash-attn {flash_attn.__version__} ready")' 2>/dev/null; then
    info "flash-attn is ready to use"
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  section "GPU summary"
  nvidia-smi -L || true
fi

if [[ -n "${HF_TOKEN_ARG}" ]]; then
  section "Hugging Face login"
  huggingface-cli login --token "${HF_TOKEN_ARG}" --add-to-git-credential || warn "HF login failed; continuing without."
fi

if [[ -n "${DL_MODEL_ID}" && -n "${DL_DEST}" ]]; then
  section "Pre-downloading model"
  # Ensure huggingface-cli is available (transformers usually brings huggingface_hub, but be explicit)
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    if [[ "${USE_UV}" -eq 1 ]] && command -v uv >/dev/null 2>&1; then
      UV_PY="--python ${VENV_DIR}/bin/python"
      uv pip ${UV_PY} install "huggingface_hub>=0.23"
    else
      python -m pip install "huggingface_hub>=0.23"
    fi
  fi
  mkdir -p "${DL_DEST}"
  huggingface-cli download "${DL_MODEL_ID}" --local-dir "${DL_DEST}" --local-dir-use-symlinks False || warn "Model download failed; continuing."
fi

if [[ "${RUN_SMOKE}" -eq 1 ]]; then
  section "Post-install smoke checks"
  python - <<'PY'
import sys
report = {}
try:
    import torch
    report['torch_version'] = getattr(torch, '__version__', 'unknown')
    report['torch_cuda_build'] = getattr(torch.version, 'cuda', 'unknown')
    report['cuda_available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        report['cuda_device_count'] = torch.cuda.device_count()
        report['devices'] = [
            {
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'cc': '.'.join(map(str, torch.cuda.get_device_capability(i)))
            }
            for i in range(torch.cuda.device_count())
        ]
        # tiny op
        x = torch.rand(1, device='cuda')
        report['cuda_op_ok'] = float(x.sin().item())
except Exception as e:
    report['torch_error'] = repr(e)

for mod in ('transformers','vllm','flash_attn'):
    try:
        __import__(mod)
        report[f'{mod}_ok'] = True
    except Exception as e:
        report[f'{mod}_ok'] = False
        report[f'{mod}_error'] = repr(e)

import json
print(json.dumps(report, indent=2))
PY
fi

section "Done"
info "Activate venv:  source ${VENV_DIR}/bin/activate"
info "Run training:   python ${REPO_ROOT}/experiments/sft/sft_train.py --config <your-config.yaml>"


