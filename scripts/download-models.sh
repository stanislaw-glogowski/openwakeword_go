#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-"$ROOT_DIR/models"}"
BASE_URL="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"

FILES=(
  "alexa_v0.1.onnx"
  "hey_jarvis_v0.1.onnx"
  "embedding_model.onnx"
  "melspectrogram.onnx"
  "silero_vad.onnx"
)

mkdir -p "$OUT_DIR"

for file in "${FILES[@]}"; do
  target="$OUT_DIR/$file"
  if [[ -f "$target" ]]; then
    echo "exists: $target"
    continue
  fi

  echo "download: $file"
  curl -L --fail --show-error --output "$target" "$BASE_URL/$file"
done

echo "models ready: $OUT_DIR"
