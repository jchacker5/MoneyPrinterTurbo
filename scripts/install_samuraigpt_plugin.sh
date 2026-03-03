#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUGIN_DIR="${ROOT_DIR}/storage/plugins/AI-Youtube-Shorts-Generator"

REPO_URL="${1:-https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator.git}"

if ! command -v git >/dev/null 2>&1; then
  echo "[warn] git not found; cannot install SamurAIGPT plugin automatically."
  exit 0
fi

mkdir -p "${ROOT_DIR}/storage/plugins"

if [[ -d "${PLUGIN_DIR}/.git" ]]; then
  echo "[info] Plugin already exists at ${PLUGIN_DIR}; pulling latest changes"
  git -C "${PLUGIN_DIR}" pull --ff-only || echo "[warn] Could not pull plugin updates"
else
  echo "[info] Cloning ${REPO_URL} into ${PLUGIN_DIR}"
  if ! git clone "${REPO_URL}" "${PLUGIN_DIR}"; then
    echo "[warn] Clone failed. You can install manually later: ${REPO_URL}"
    exit 0
  fi
fi

if [[ -f "${PLUGIN_DIR}/requirements.txt" ]]; then
  echo "[info] Installing plugin Python requirements"
  if command -v pip >/dev/null 2>&1; then
    pip install -r "${PLUGIN_DIR}/requirements.txt" || echo "[warn] plugin requirements install failed"
  else
    echo "[warn] pip not found; skipping plugin requirements"
  fi
fi

echo "[info] Plugin installer completed"
