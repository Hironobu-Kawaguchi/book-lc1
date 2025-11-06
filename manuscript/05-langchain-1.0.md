# LangChain 1.0：アプリケーション高速開発レイヤー

## LangChain 1.0 とは

LangChainはエンドツーエンド構築を簡潔にする抽象とツール群を提供します。
LangChain は、LLM アプリケーションの開発を迅速化するための高レベルライブラリです。
プロトタイピングから本番運用まで、開発者が簡潔なコードで複雑な機能を実現できるように設計されています。

LangChain の強みは以下の通りです。

- **高レベル抽象**: エージェント、チェーン、RAG などを数行で構築
- **豊富なインテグレーション**: 主要な LLM プロバイダ、ベクトル DB、外部 API に対応
- **プロトタイピングの高速化**: 素早く試行錯誤できる
- **標準化されたインターフェース**: 異なる LLM 間での切り替えが容易

一方で、高レベル抽象であるがゆえに、細かい制御が難しい場合があります。
複雑なワークフローが必要な場合は、LangGraph の方が適しています。

## LangChain 1.0 の主要な変更点

主要な変更点はcreate_agent APIとcontent_blocks形式と標準化されたエージェント抽象です。

### create_agent API

`create_agent` は、エージェントを構築するための統一的な API です。
従来の `initialize_agent` や `AgentExecutor` に代わる、よりシンプルなインターフェースです。

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def calculate(expression: str) -> str:
    """数式を計算します。"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"エラー: {e}"

llm = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(llm, [calculate])

response = agent.invoke({"messages": [{"role": "user", "content": "100 * 25 を計算して"}]})
```

`create_react_agent` は ReAct（Reasoning and Acting）パターンを実装したエージェントを簡単に構築できます。

### content_blocks 形式

`content_blocks` は、メッセージの内容を柔軟に表現する新しい形式です。
テキストだけでなく、画像、ツール呼び出し、ツール実行結果など、多様なコンテンツを統一的に扱えます。

```python
from langchain_core.messages import HumanMessage

message = HumanMessage(
    content=[
        {"type": "text", "text": "この画像を分析してください"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    ]
)
```

これにより、マルチモーダルなエージェント構築が容易になります。

### 標準化されたエージェント抽象

LangChain 1.0 では、エージェントのインターフェースが標準化されました。
すべてのエージェントは `invoke` メソッドを持ち、同じ方法で呼び出せます。
これにより、エージェントの切り替えやテストが容易になります。

## 実装レベル1：簡易FAQエージェント

実装レベル1ではcreate_agentで簡易FAQエージェントを構築する例を示します。

### FAQ データベースを検索するツール

社内の FAQ データベースを検索するツールを定義します。

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# サンプルFAQデータ
faq_database = {
    "営業時間": "営業時間は平日9時から18時までです。",
    "返品": "商品到着後14日以内であれば返品可能です。",
    "配送": "通常、注文から3営業日以内に配送いたします。"
}

@tool
def search_faq(query: str) -> str:
    """FAQデータベースからキーワードに関連する情報を検索します。"""
    for key, value in faq_database.items():
        if key in query:
            return value
    return "該当する情報が見つかりませんでした。"

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_react_agent(llm, [search_faq])

# エージェントを実行
response = agent.invoke({
    "messages": [HumanMessage(content="営業時間を教えてください")]
})

print(response["messages"][-1].content)
```

このエージェントは、ユーザーの質問を理解し、FAQ データベースを検索して回答を生成します。

## 実装レベル2：RAG統合とツール呼び出し

実装レベル2ではRAG統合とツール呼び出し付きエージェントを構築します。

### ベクトル検索を使った RAG エージェント

ドキュメントをベクトル化して検索し、LLM に渡すエージェントを構築します。

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# サンプルドキュメントを作成
documents = [
    Document(page_content="LangChainはLLMアプリケーション開発のためのフレームワークです。", metadata={"source": "doc1"}),
    Document(page_content="LangGraphは複雑なワークフローを構築するためのツールです。", metadata={"source": "doc2"}),
    Document(page_content="RAGは検索と生成を組み合わせた手法です。", metadata={"source": "doc3"}),
]

# ベクトルストアを構築
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

@tool
def search_documents(query: str) -> str:
    """ドキュメントデータベースから関連情報を検索します。"""
    results = vectorstore.similarity_search(query, k=2)
    return "\n\n".join([doc.page_content for doc in results])

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_react_agent(llm, [search_documents])

# エージェントを実行
response = agent.invoke({
    "messages": [HumanMessage(content="LangChainとLangGraphの違いを教えて")]
})

print(response["messages"][-1].content)
```

このエージェントは、ベクトル検索を使って関連文書を取得し、それを基に回答を生成します。

### 複数ツールを組み合わせたエージェント

検索と計算の両方を行えるエージェントを構築します。

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

@tool
def web_search(query: str) -> str:
    """Web検索を実行します。"""
    # 実際にはAPIを呼び出すが、ここでは固定値を返す
    return f"{query}に関する最新情報: サンプルの検索結果です。"

@tool
def calculate(expression: str) -> str:
    """数式を計算します。"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"エラー: {e}"

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_react_agent(llm, [web_search, calculate])

# エージェントを実行
response = agent.invoke({
    "messages": [HumanMessage(content="Pythonの最新バージョンを調べて、そのバージョン番号を10倍してください")]
})

print(response["messages"][-1].content)
```

エージェントは複数のツールを適切に選択し、組み合わせてタスクを実行します。

## LCEL（LangChain Expression Language）

LCEL は、チェーンを簡潔に記述するための構文です。
`|` 演算子を使って処理を連結できます。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_template("次の質問に答えてください: {question}")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

response = chain.invoke({"question": "AIとは何ですか？"})
print(response)
```

LCEL を使うことで、プロンプト、LLM、パーサーを直感的に連結できます。

## 運用観点：プロトタイピングから本番化

運用観点としてプロトタイピングから本番化とモニタリングとメトリクス設計を説明します。

### プロトタイピング

LangChain は高速なプロトタイピングに最適です。
Jupyter Notebook や Python スクリプトで素早く試行錯誤できます。

プロトタイピング時のポイント：
- シンプルなツールから始める
- ログを出力して挙動を確認
- 少量のテストデータで検証

### 本番化の考慮事項

本番環境に移行する際の考慮事項：

1. **エラーハンドリング**: ツールの実行失敗やタイムアウトに対応
2. **レート制限**: LLM API の呼び出し回数を制御
3. **コスト管理**: トークン使用量を監視
4. **セキュリティ**: API キーの管理、入力のサニタイズ

### モニタリングとメトリクス

LangSmith を使って、エージェントの実行をモニタリングできます。

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-project"
```

トレースを有効にすると、以下の情報が記録されます：

- LLM の入出力
- ツールの呼び出し履歴
- 実行時間
- エラー情報

これらの情報を分析することで、ボトルネックや改善点を特定できます。

## 注意点とトレードオフ

注意点として高レベル抽象のカスタマイズ限界とブラックボックス化リスクを挙げます。

### 高レベル抽象のカスタマイズ限界

LangChain の高レベル API は便利ですが、カスタマイズに限界があります。
例えば、エージェントのループ回数を制限したい、特定の条件で処理を分岐させたい、といった場合は、LangGraph を使う方が適しています。

### ブラックボックス化のリスク

高レベル抽象を使うと、内部で何が起きているのか見えにくくなります。
デバッグが困難になることがあるため、トレーシングを有効にして挙動を確認することが重要です。

### パフォーマンスの考慮

LangChain の抽象層は便利ですが、オーバーヘッドが発生します。
高速な応答が必要な場合は、直接 LLM の API を呼び出す方が効率的な場合もあります。

## LangChain と LangGraph の使い分け

- **LangChain を使う場合**:
  - プロトタイピングや PoC
  - シンプルなエージェント構築
  - 標準的なユースケース（FAQ、RAG など）

- **LangGraph を使う場合**:
  - 複雑なワークフローが必要
  - 長時間実行やヒューマンインザループが必要
  - マルチエージェント協調
  - 細かい制御が必要

## 本章のまとめ

本章では、LangChain 1.0 の主要な機能、`create_agent` API、RAG との統合、LCEL、運用観点を解説しました。
LangChain は高速なプロトタイピングと標準的なユースケースに適していますが、複雑なワークフローには LangGraph が必要です。
次章では、RAG との統合パターンをより詳しく見ていきます。
