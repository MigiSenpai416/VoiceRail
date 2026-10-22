#!/usr/bin/env bash
# VoiceRail on a rented Linux GPU pod (RunPod, Vast.ai, Lambda, Paperspace, ...).
# Linux only; on Windows use voicerail.bat.
# Run it from a Jupyter terminal or a notebook cell; the first run installs
# everything, later runs start in seconds. Arguments are passed to
# `voicerail`, the default is the web UI reachable from outside the pod:
#   ./pod.sh                          web UI on 0.0.0.0:8080 + public share link
#   VOICERAIL_SHARE=0 ./pod.sh        no share link (use the pod's port proxy)
#   ./pod.sh download                 install and fetch the base models only
#   ./pod.sh train MyVoice --data /workspace/recordings
# Everything slow to recreate (Python, packages, models, training sessions,
# outputs) is kept on the persistent volume so a pod restart does not lose it.
set -euo pipefail
cd "$(dirname "$0")"

volume="${VOICERAIL_VOLUME:-}"
if [ -z "$volume" ]; then
  # /workspace is the persistent volume on RunPod and Vast.ai; otherwise keep
  # everything next to the repository.
  if [ -d /workspace ] && [ -w /workspace ]; then volume=/workspace; else volume="$PWD"; fi
fi
export VOICERAIL_HOME="${VOICERAIL_HOME:-$volume/voicerail-data}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$volume/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$volume/.cache/uv-python}"
# Volumes are usually a different filesystem than the uv cache: copy instead
# of hard-linking so uv does not warn on every install.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export VOICERAIL_HOST="${VOICERAIL_HOST:-0.0.0.0}"
export VOICERAIL_PORT="${VOICERAIL_PORT:-8080}"
export VOICERAIL_SHARE="${VOICERAIL_SHARE:-1}"
export VOICERAIL_NO_BROWSER=1
mkdir -p "$VOICERAIL_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

echo "VoiceRail: data in $VOICERAIL_HOME"
if [ $# -eq 0 ]; then
  if [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "RunPod port proxy: https://${RUNPOD_POD_ID}-${VOICERAIL_PORT}.proxy.runpod.net (expose HTTP port ${VOICERAIL_PORT} in the pod template)"
  fi
  set -- web
fi
exec ./voicerail.sh "$@"
