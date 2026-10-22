#!/usr/bin/env bash
# VoiceRail launcher for Linux and macOS.
# First run installs everything (uv, Python 3.12, PyTorch, ...), later runs
# just start the app. Arguments are passed to `voicerail`, e.g.
#   ./voicerail.sh                 start the web UI
#   ./voicerail.sh web --host 0.0.0.0   web UI reachable from other machines
#   ./voicerail.sh convert song.mp3 -m MyVoice
# Set VOICERAIL_TORCH=cuda|cuda-legacy|rocm|cpu to override GPU detection.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Python package manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

variant="${VOICERAIL_TORCH:-}"
if [ -z "$variant" ]; then
  if [ "$(uname -s)" = "Darwin" ]; then
    variant=cpu
  elif command -v nvidia-smi >/dev/null 2>&1; then
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
      | head -n1 | cut -d. -f1 || true)
    if [ -n "${driver:-}" ] && [ "$driver" -ge 580 ]; then
      variant=cuda
    elif [ -n "${driver:-}" ] && [ "$driver" -ge 525 ]; then
      variant=cuda-legacy
    else
      variant=cpu
      echo "NVIDIA driver is too old for GPU acceleration; update it for much faster processing."
    fi
  elif command -v rocminfo >/dev/null 2>&1; then
    variant=rocm
  else
    variant=cpu
  fi
fi

echo "VoiceRail: using the '$variant' PyTorch build."
uv sync --frozen --no-dev --extra "$variant"
exec uv run --no-sync voicerail "$@"
