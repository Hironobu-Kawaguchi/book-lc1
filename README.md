# エージェント＆RAG時代の実践 — 原稿レポジトリ

このレポジトリは書籍原稿とビルド周りに加えてチュートリアル用のコードを `tutorial/` 配下で管理します。

## 構成

```
.
├─ manuscript/           # 原稿（章ごと、1文1行）
├─ images/               # 図版・表紙（Git LFS 管理）
├─ scripts/              # 補助スクリプト（sanitize 等）
├─ build/                # 生成物（.gitignore 済み）
├─ .github/workflows/    # CI（build.yml / release.yml）
├─ metadata.yml          # 書籍メタデータ（title/author/lang）
├─ epub.css              # EPUB スタイル
├─ Makefile              # ローカルビルド入口
└─ README.md
```

## 事前準備（原稿ビルド）

- macOS: `brew install pandoc`
- コードレポの環境構築は `uv` 前提です（Poetry/Dockerは使用しません）。
- 他OS: Pandoc をインストールしてください。

## ビルド

```bash
make build
```

実行内容: `scripts/sanitize_manuscript.sh` で TODO 行を除去 → Pandoc で `build/book.epub` を生成します。

手動ビルド例:

```bash
bash scripts/sanitize_manuscript.sh
pandoc --from=gfm --standalone \
  --metadata-file=metadata.yml --css=epub.css \
  --highlight-style=kate --toc -o build/book.epub build/sanitized/*.md
```

## チュートリアルコードの環境（uv）

チュートリアルコードは `tutorial/` ディレクトリ配下にあります。
uv を使い、`tutorial/` 内で仮想環境と依存関係をセットアップします。

例（`tutorial/` で実行）:

```bash
# uv の導入（macOS/Homebrew）
brew install uv

# Python 3.12 系を用意（任意。ローカルに無ければ）
uv python install 3.12

# 仮想環境を作成して有効化
uv venv -p 3.12
source .venv/bin/activate

# 依存解決（pyproject.toml がある前提）
uv sync
```

サンプルの実行例:

```bash
# API キー設定（初回のみ）
cp .env.sample .env && $EDITOR .env

# LangGraph の最小例（第4章）
uv run python ch04/level1_graph_greeting.py

# LangChain の最小エージェント例（第5章、APIキーで自動選択）
uv run python ch05/level1_agent_faq.py

# LangChain: Middleware でPIIマスク
uv run python ch05/level2_agent_middleware.py

# LangChain: ToolStrategyで構造化レスポンス
uv run python ch05/level2_agent_structured_output.py

# LangChain: content_blocks を確認
uv run python ch05/feature_content_blocks.py
```
```

## ライセンス

- コード: MIT（`LICENSE`）
- コンテンツ: CC BY 4.0（`LICENSE-CONTENT`）

## ガイドライン（抜粋）

- Prose: 1文1行。
- 章ファイルは `00-`, `01-` の番号接頭辞で順序固定。
- 画像は `images/` に配置し、必要に応じて LFS 管理。
