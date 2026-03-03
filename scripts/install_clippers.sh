#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

have() {
  command -v "$1" >/dev/null 2>&1
}

install_pip_pkg() {
  local pkg="$1"
  if python - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("${pkg}")
PY
  then
    echo "[ok] Python package '${pkg}' already installed"
  else
    echo "[info] Installing Python package '${pkg}'"
    pip install "${pkg}" || echo "[warn] Failed to install '${pkg}'"
  fi
}

echo "[info] Installing clipper dependencies in ${ROOT_DIR}"

if have yt-dlp; then
  echo "[ok] yt-dlp detected"
else
  echo "[info] Installing yt-dlp"
  pip install yt-dlp || echo "[warn] Could not install yt-dlp via pip"
fi

if have ffmpeg; then
  echo "[ok] ffmpeg detected"
else
  echo "[warn] ffmpeg not found in PATH. Install it via your package manager."
fi

if have ffprobe; then
  echo "[ok] ffprobe detected"
else
  echo "[warn] ffprobe not found in PATH (usually provided by ffmpeg package)"
fi

install_pip_pkg faster_whisper
install_pip_pkg whisper
install_pip_pkg scenedetect

if have auto-editor; then
  echo "[ok] auto-editor detected"
else
  echo "[info] Installing auto-editor"
  pip install auto-editor || echo "[warn] Could not install auto-editor"
fi

if [[ "${INSTALL_SAMURAIGPT_PLUGIN:-0}" == "1" ]]; then
  echo "[info] Installing optional SamurAIGPT plugin"
  bash "${ROOT_DIR}/scripts/install_samuraigpt_plugin.sh" || true
else
  echo "[info] Skipping optional SamurAIGPT plugin install (set INSTALL_SAMURAIGPT_PLUGIN=1 to enable)"
fi

echo "[info] Clipper dependency installation finished"
