#!/usr/bin/env bash
# One-line VoiceRail setup for a Linux GPU pod (RunPod, Vast.ai, Lambda, ...):
#
#   curl -fsSL https://raw.githubusercontent.com/MigiSenpai416/VoiceRail/main/install.sh | bash
#
# Clones (or updates) the repository on the persistent volume, installs uv,
# Python, PyTorch and the app, then starts the web UI on 0.0.0.0:9999.
# Anything after `bash -s --` is passed to pod.sh / voicerail, e.g.
#   ... | bash -s -- download                  install and fetch base models only
#   ... | bash -s -- train MyVoice --data /workspace/recordings
# Environment: VOICERAIL_REPO, VOICERAIL_BRANCH, VOICERAIL_DIR, VOICERAIL_VOLUME.
set -euo pipefail

main() {
  local repo="${VOICERAIL_REPO:-https://github.com/MigiSenpai416/VoiceRail.git}"
  local branch="${VOICERAIL_BRANCH:-main}"
  local volume="${VOICERAIL_VOLUME:-}"
  if [ -z "$volume" ]; then
    if [ -d /workspace ] && [ -w /workspace ]; then volume=/workspace; else volume="$HOME"; fi
  fi
  local dir="${VOICERAIL_DIR:-$volume/VoiceRail}"

  if ! command -v git >/dev/null 2>&1; then
    echo "Installing git..."
    if command -v apt-get >/dev/null 2>&1; then
      local sudo=""
      [ "$(id -u)" -ne 0 ] && sudo="sudo"
      $sudo apt-get update -qq </dev/null && $sudo apt-get install -y -qq git </dev/null
    else
      echo "git is required; install it and run this again." >&2
      exit 1
    fi
  fi

  if [ -d "$dir/.git" ]; then
    echo "VoiceRail: updating $dir"
    git -C "$dir" pull --ff-only </dev/null
  else
    echo "VoiceRail: cloning $repo ($branch) into $dir"
    git clone --branch "$branch" "$repo" "$dir" </dev/null
  fi
  cd "$dir"
  chmod +x pod.sh voicerail.sh
  export VOICERAIL_VOLUME="$volume"
  exec ./pod.sh "$@" </dev/null
}

main "$@"
