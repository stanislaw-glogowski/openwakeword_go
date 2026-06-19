#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-"$ROOT_DIR/runtime"}"
VERSION="1.26.0"
BASE_URL="https://github.com/microsoft/onnxruntime/releases/download/v$VERSION"

os="$(uname -s)"
arch="$(uname -m)"

case "$os:$arch" in
  Darwin:arm64)
    archive="onnxruntime-osx-arm64-$VERSION.tgz"
    library="libonnxruntime.dylib"
    ;;
  Darwin:x86_64)
    archive="onnxruntime-osx-x86_64-$VERSION.tgz"
    library="libonnxruntime.dylib"
    ;;
  Linux:x86_64)
    archive="onnxruntime-linux-x64-$VERSION.tgz"
    library="libonnxruntime.so"
    ;;
  Linux:aarch64 | Linux:arm64)
    archive="onnxruntime-linux-aarch64-$VERSION.tgz"
    library="libonnxruntime.so"
    ;;
  MINGW*:x86_64 | MSYS*:x86_64 | CYGWIN*:x86_64)
    archive="onnxruntime-win-x64-$VERSION.zip"
    library="onnxruntime.dll"
    ;;
  *)
    echo "unsupported platform: $os $arch" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"

target="$OUT_DIR/$library"
if [[ -f "$target" ]]; then
  echo "exists: $target"
  exit 0
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

echo "download: $archive"
curl -L --fail --show-error --output "$tmp_dir/$archive" "$BASE_URL/$archive"

case "$archive" in
  *.tgz)
    tar -xzf "$tmp_dir/$archive" -C "$tmp_dir"
    found="$(find "$tmp_dir" -type f -name "$library" | head -n 1)"
    ;;
  *.zip)
    unzip -q "$tmp_dir/$archive" -d "$tmp_dir"
    found="$(find "$tmp_dir" -type f -name "$library" | head -n 1)"
    ;;
esac

if [[ -z "${found:-}" ]]; then
  echo "runtime library not found in archive: $library" >&2
  exit 1
fi

cp "$found" "$target"
echo "runtime ready: $target"
