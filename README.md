# Kindle 向け Markdown/Pandoc 執筆テンプレート

GitHub と Pandoc を用いて、Markdown 原稿から高品質な EPUB3 をビルドし、GitHub Actions で自動化・配布（Releases）するためのテンプレートです。

## リポジトリ構成

```
.
├─ .github/workflows/
│  ├─ build.yml          # main への push で EPUB を自動ビルド
│  └─ release.yml        # v* タグ push で Release に EPUB を添付
├─ manuscript/           # 原稿（1 文 1 行推奨）
│  ├─ 00-frontmatter.md
│  ├─ 01-introduction.md
│  └─ 02-chapter-one.md
├─ images/               # 図版・表紙（Git LFS 管理推奨）
├─ build/                # ビルド成果物（.gitignore 済み）
├─ metadata.yml          # 書籍メタデータ（title, author, lang 等）
├─ epub.css              # EPUB 用カスタム CSS
├─ .gitattributes        # 画像などを Git LFS で追跡
├─ .gitignore
└─ README.md
```

## ワークフロー概要

- Docs-as-Code 方針で原稿を Markdown（GFM）で管理
- Pandoc で EPUB3 を生成（`epub.css` を適用）
- GitHub Actions で CI ビルド（push 時に artifact として EPUB を保存）
- タグ `v*` を push すると GitHub Releases に EPUB を自動添付

## 1 文 1 行（One Sentence Per Line）

Git の diff を読みやすくするため、散文でも「1 文 1 行」を徹底してください。レビュー効率が飛躍的に向上します。

## 画像と Git LFS

大きな画像はリポジトリを肥大化させます。以下の設定を済ませてから画像を追加してください。

```bash
brew install git-lfs           # 未インストールの場合
git lfs install
# 追跡ルールは .gitattributes に用意済み（jpg / png / svg / gif）
```

## メタデータと表紙

- `metadata.yml` を編集して、`title`、`author`、`lang` 等を設定
- 表紙を用意したら `images/cover.jpg` を追加し、`metadata.yml` の `cover-image` を有効化

## ローカルビルド

事前に Pandoc をインストールしてください（例：`brew install pandoc`）。

```bash
# 1) 中間ディレクトリと成果物ディレクトリを作成
mkdir -p build/sanitized

# 2) TODO 行の除去（WIP メモをビルドから除外）
bash scripts/sanitize_manuscript.sh

# 3) Pandoc で EPUB を生成
pandoc \
  --from=gfm \
  --standalone \
  --metadata-file=metadata.yml \
  --output=build/book.epub \
  --toc \
  --css=epub.css \
  --syntax-highlighting=kate \
  build/sanitized/*.md
```

または Makefile を使う場合：

```bash
make build
```

## GitHub Actions

- `build.yml`: `main` への push で自動ビルドし、Actions の artifact に `book.epub` を保存
- `release.yml`: `v*` タグの push をトリガーにビルドし、対応する GitHub Release に `book.epub` を添付

## KDP へのアップロード

KDP は EPUB の手動アップロードのみをサポートします。出版時は GitHub Releases から最新版の `book.epub` をダウンロードし、KDP の Bookshelf からアップロードしてください。

## ライティング TIPS

- GFM 記法を基本に、見出しは章先頭を `#`、節は `##` として階層化
- 各章ファイルは `00-`, `01-` のような番号付き接頭辞で順序を安定化
- 画像は `images/` に置き、本文から相対パスで参照（例：`![図1](images/ch01-diagram.png)`）

## タグ付けとリリース

```bash
git tag v1.0.0
git push origin v1.0.0
```

数十秒後に GitHub の Releases ページに `book.epub` が添付されます。

## ライセンス

本リポジトリはデュアルライセンスです。

- コード（`scripts/`, `.github/`, `Makefile`, `epub.css` など）: MIT License（`LICENSE`）
- コンテンツ（`manuscript/` の本文、`images/` の画像）: CC BY 4.0（`LICENSE-CONTENT`）
- 例外素材やクレジットは `ATTRIBUTIONS.md` に記載します。

KDPで販売するEPUBの販売権は著者に帰属します（リポジトリの公開ライセンスとは独立）。
