# Repository Guidelines

## Project Structure & Module Organization

```
.
├─ manuscript/           # 原稿（章ごと、1文1行）
├─ images/               # 図版・表紙（Git LFS 管理）
├─ scripts/              # 補助スクリプト（sanitize 等）
├─ build/                # 生成物（.gitignore 済み）
├─ .github/workflows/    # CI（build.yml / release.yml）
├─ metadata.yml          # 書籍メタデータ（title/author/lang/cover）
├─ epub.css              # EPUB スタイル
├─ Makefile              # ローカルビルド入口
└─ README.md
```

## Build, Test, and Development Commands

- Prerequisite: `brew install pandoc`
- Build locally: `make build`
  - 実行内容: `scripts/sanitize_manuscript.sh` で TODO 行を除去 → Pandoc で `build/book.epub` 生成。
- Manual build (example):
  ```bash
  bash scripts/sanitize_manuscript.sh
  pandoc --from=gfm --standalone \
    --metadata-file=metadata.yml --css=epub.css \
    --syntax-highlighting=kate --toc -o build/book.epub build/sanitized/*.md
  ```
- CI: `main` へ push で artifact 作成。`git tag v1.0.0 && git push origin v1.0.0` で Release に EPUB 添付。

## Coding Style & Naming Conventions

- Prose: 1 sentence per line（1文1行）。
- Files: 章は `00-`, `01-` の番号接頭辞で順序を固定。見出しは `#`, `##` を階層的に使用。
- Markdown: GFM 準拠。不要な HTML は避ける。画像は `images/` に配置し `![alt](images/xxx.png)` で参照。
- CSS: 体裁調整は `epub.css` に集約（インラインスタイルは原則禁止）。

## Testing Guidelines

- 現状ユニットテストは未導入。ビルド成功と目視確認を必須とする。
- 推奨チェック（任意）: EPUB ビューアでのレイアウト確認、リンク切れ・体裁崩れの点検。

## Commit & Pull Request Guidelines

- Flow: GitHub Flow（`main` は常にリリース可能）。
- Branch names: `feat/…`, `fix/…`, `docs/…`。
- Commits: 短く具体的に（例: `feat: add 02-chapter-one skeleton`）。
- PR: 変更概要、背景、影響範囲、ビルド結果を記載し、必要なら `Closes #123` を付与。
- Gates: TODO 行を残さない／LFS 対象の画像のみ追加／CI green を確認。

## Security & Configuration Tips

- 秘密情報（KDP 資格情報等）をコミットしない。
- 大きな画像や表紙は LFS（`.gitattributes` 済）。`build/` 配下は成果物のためコミット禁止。
 - ライセンス: コードは MIT（`LICENSE`）、コンテンツは CC BY 4.0（`LICENSE-CONTENT`）。外部素材は `ATTRIBUTIONS.md` に記載。
