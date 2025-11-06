#!/usr/bin/env bash
set -euo pipefail

# manuscript/*.md を走査して、TODO 系メモを除去したコピーを build/sanitized/ に生成する。
# 対象: 行頭が `TODO`、`TODO:`、`TODO [..]:` などの簡易パターン。

SRC_DIR="manuscript"
OUT_DIR="build/sanitized"

mkdir -p "$OUT_DIR"

for f in "$SRC_DIR"/*.md; do
  base=$(basename "$f")
  # `^
  #  \s*TODO(\s|\[|:)` にマッチする行を削除。
  # BSD sed 互換のため -E を使用。
  sed -E '/^\s*TODO(\s|\[|:)/d' "$f" > "$OUT_DIR/$base"
done

echo "Sanitized markdown written to $OUT_DIR" >&2

