# エコシステムの変遷と1.0版の意味

## LangChain と LangGraph の歴史

LangChainとLangGraphの歴史と1.0に至る背景を俯瞰します。

### LangChain の誕生と成長

LangChain は 2022 年に Harrison Chase によって開始されたオープンソースプロジェクトです。
当初は LLM を使ったアプリケーション開発を簡素化するための薄いラッパーとして設計されました。
プロンプトテンプレート、チェーン（複数のステップを連結する仕組み）、メモリ（会話履歴の管理）などの抽象を提供し、開発者が LLM を簡単に利用できるようにしました。

LangChain は急速に成長し、コミュニティからの貢献により、多様なインテグレーション（ベクトルDB、外部 API、ツール）が追加されました。
しかし、成長に伴いコードベースが複雑化し、API が頻繁に変更されるという課題が生じました。

### LangGraph の登場

LangGraph は 2023 年に LangChain チームから派生したプロジェクトとして登場しました。
LangChain の高レベル抽象では表現しきれない、複雑なステートフルワークフローを扱うために設計されました。
グラフ構造でワークフローを表現し、ノード間の状態遷移を明示的に制御できる点が特徴です。

LangGraph は長時間実行、チェックポイント、ヒューマンインザループといった運用上の要件に対応しています。
マルチエージェント協調やシミュレーションなど、高度なユースケースにも適用可能です。

## 1.0 版の登場とその意義

1.0ではエージェントの抽象とAPIの安定化と運用面の強化が進みました。

### API の安定化

LangChain と LangGraph はともに、0.x 系から 1.0 系への移行を果たしました。
1.0 の最も重要な意義は、API の安定化です。
0.x 系ではマイナーバージョンアップごとに破壊的変更が入ることが多く、本番運用には不安がありました。
1.0 以降は Semantic Versioning に従い、メジャーバージョンが変わらない限り互換性が保証されます。

### エージェントの抽象化

LangChain 1.0ではcreate_agentが中心APIとなりコンテンツ表現がcontent_blocksに整理されました。
`create_agent` API は、エージェントの構築を統一的なインターフェースで行えるようにします。
従来は複数の異なる API（AgentExecutor、initialize_agent など）が存在しましたが、1.0 では整理されました。

`content_blocks` は、メッセージの内容をテキストだけでなく、画像、ツール呼び出し、ツール実行結果など、多様な形式で表現できる仕組みです。
これにより、マルチモーダルなエージェント構築が容易になりました。

### 運用面の強化

1.0 では、トレーシング、メトリクス収集、エラーハンドリングといった運用面の機能が強化されました。
LangSmith（LangChain 公式の観測性プラットフォーム）との統合が進み、本番運用時のデバッグや性能分析が容易になりました。

## v0 系から 1.0 系への移行

ミニ演習としてv0系コードを1.0系へリファクタリングする観点を解説します。

### 主な変更点

v0 系から 1.0 系への移行では、以下の点に注意が必要です。

1. **エージェント構築 API の変更**
   - 旧: `initialize_agent` や `AgentExecutor`
   - 新: `create_agent` または `create_react_agent`

2. **メッセージ形式の変更**
   - 旧: `content` は文字列のみ
   - 新: `content_blocks` でテキスト、画像、ツール呼び出しを統一的に扱う

3. **ツール定義の変更**
   - 旧: `BaseTool` を継承してクラスを作成
   - 新: `@tool` デコレータで関数を装飾（より簡潔）

4. **チェーンの表現**
   - 旧: `LLMChain` や `SequentialChain`
   - 新: LCEL（LangChain Expression Language）で `|` 演算子を使って連結

### リファクタリング例

v0 系のコード：

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

def search_function(query: str) -> str:
    return f"検索結果: {query}"

tools = [Tool(name="Search", func=search_function, description="検索を実行")]
llm = ChatOpenAI(model="gpt-4")
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

1.0 系のコード：

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def search(query: str) -> str:
    """検索を実行します。"""
    return f"検索結果: {query}"

llm = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(llm, [search])
```

より簡潔で読みやすくなりました。

## 互換性と移行戦略

互換性と移行手順とテスト戦略を明確にして安全に移行します。

### 段階的な移行

一度にすべてのコードを書き換えるのはリスクが高いため、段階的に移行します。

1. **依存パッケージのバージョンアップ**
   - `langchain` と `langgraph` を 1.0 系にアップデート
   - 互換性レイヤーが提供されているため、一部の v0 系 API は引き続き動作

2. **新規開発から 1.0 API を採用**
   - 新しく作成するエージェントやツールは 1.0 の API を使用
   - 既存コードは動作を確認しながら順次移行

3. **テストカバレッジの確保**
   - 移行前にテストを書いて、既存機能が壊れないことを確認
   - 特に、ツール呼び出しや状態管理の挙動をテスト

### 非推奨 API の確認

LangChain 1.0 では、非推奨（deprecated）となった API に警告が表示されます。
ログを確認し、警告が出た箇所から優先的に移行します。

### ドキュメントとコミュニティの活用

公式ドキュメントには移行ガイドが用意されています。
LangChain の GitHub リポジトリや Discord コミュニティで質問することも有効です。

## 本章のまとめ

本章では、LangChain と LangGraph の歴史、1.0 版の意義、v0 系から 1.0 系への移行方法を解説しました。
1.0 では API の安定化と運用面の強化が進み、本番運用に適した成熟したエコシステムになりました。
次章では、LangGraph 1.0 の具体的な機能と実装パターンを詳しく見ていきます。

