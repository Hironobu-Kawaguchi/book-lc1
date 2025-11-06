# Tutorial Code (uv)

本ディレクトリには書籍のチュートリアル用ソースコードを配置します。
Python 環境は `uv` を前提とします。

## セットアップ

```bash
cd tutorial
brew install uv # 未導入の場合
uv python install 3.12
uv venv -p 3.12
source .venv/bin/activate
uv sync

# APIキーの設定（初回のみ）
cp .env.sample .env
# .env を開いて OPENAI_API_KEY または ANTHROPIC_API_KEY を設定
```

## 実行例

```bash
# LangGraph: 最小グラフ（第4章）
uv run python ch04/level1_graph_greeting.py

# LangChain: FAQエージェント（第5章、APIキーで自動選択）
uv run python ch05/level1_agent_faq.py

# LangChain: Middleware でPIIマスク（第5章）
uv run python ch05/level2_agent_middleware.py

# LangChain: ToolStrategyで構造化レスポンス（第5章）
uv run python ch05/level2_agent_structured_output.py

# LangChain: content_blocks を表示（第5章）
uv run python ch05/feature_content_blocks.py
```

## 構成

```
tutorial/
├─ pyproject.toml         # uv/PEP 621 メタデータ
├─ .env.sample            # APIキーの雛形（OPENAI_API_KEY/ANTHROPIC_API_KEY）
├─ ch04/                   # 第4章: LangGraph 最小例 等
└─ ch05/                   # 第5章: LangChain v1 チュートリアル群
```
