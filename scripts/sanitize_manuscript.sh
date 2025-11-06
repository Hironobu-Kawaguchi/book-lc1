#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
IN_DIR="$ROOT_DIR/manuscript"
OUT_DIR="$ROOT_DIR/build/sanitized"

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/*.md || true

shopt -s nullglob
files=("$IN_DIR"/*.md)
IFS=$'\n' sorted=($(printf '%s\n' "${files[@]}" | sort))

for f in "${sorted[@]}"; do
  base=$(basename "$f")
  # Remove lines that begin with TODO or contain TODO as a standalone marker.
  # Preserve 1 sentence per line style as-is.
  sed -E '/^\s*TODO\b.*/d' "$f" > "$OUT_DIR/$base"
done

echo "Sanitized markdown written to $OUT_DIR"

