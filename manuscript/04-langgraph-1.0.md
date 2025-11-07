# LangGraph 1.0：グラフとオーケストレーション基盤

## LangGraph とは何か

LangGraphは長時間実行と状態管理とマルチエージェント協調に強みを持つ基盤です。
LangChain が提供する高レベル抽象では表現しきれない、複雑なワークフローを構築するために設計されました。
グラフ構造を用いることで、エージェントの挙動を視覚的に理解しやすく、デバッグや保守も容易になります。

LangGraph の特徴は以下の通りです。

- **明示的なワークフロー定義**: ノードとエッジで処理の流れを表現
- **ステートフル**: 状態をグラフ全体で共有し、各ノードで更新
- **長時間実行**: チェックポイントを使って中断・再開が可能
- **ヒューマンインザループ**: 人間の承認を待つフローを簡単に実装
- **マルチエージェント協調**: 複数のエージェントが協調して問題を解決

## グラフの基本要素

ノードとエッジと状態とストアを用いてワークフローを明示的に表現します。

### ノード（Node）

ノードは処理の単位です。
各ノードは Python 関数として定義され、状態を受け取り、処理を行い、状態を更新します。
ノードには以下のような役割があります。

- **LLM 呼び出し**: ユーザーの入力を受け取り、LLM に問い合わせて応答を生成
- **ツール実行**: 検索や計算などの外部ツールを実行
- **データ変換**: 状態の加工やフィルタリング
- **分岐判断**: 次にどのノードに進むかを決定

### エッジ（Edge）

エッジはノード間の遷移を表します。
エッジには以下の種類があります。

- **通常のエッジ**: ノード A から B へ無条件に遷移
- **条件付きエッジ**: 状態に応じて遷移先を動的に決定
- **開始エッジ**: グラフの最初のノードを指定
- **終了エッジ**: グラフの終了を示す

### 状態（State）

状態はグラフ全体で共有されるデータ構造です。
Pydantic のモデルや TypedDict を使って定義します。
各ノードは状態を読み取り、更新します。

状態には以下のような情報が含まれます。

- ユーザーの入力メッセージ
- LLM の応答
- ツールの実行結果
- 中間計算結果
- エラー情報

### ストア（Store）

ストアは永続化された状態です。
チェックポイント機能を使うと、グラフの実行途中で状態を保存し、後から再開できます。
これにより、長時間実行や非同期処理が可能になります。

## 実装レベル1：最小グラフ

実装レベル1では単一ノードの挨拶から返答までの最小グラフを示します。

### 最小グラフの例

ユーザーの入力を受け取り、挨拶を返す最小限のグラフを構築します。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    message: str
    response: str

def greet_node(state: State) -> State:
    """挨拶を生成するノード"""
    user_message = state["message"]
    state["response"] = f"こんにちは！{user_message}と言いましたね。"
    return state

# グラフの構築
graph = StateGraph(State)
graph.add_node("greet", greet_node)
graph.add_edge(START, "greet")
graph.add_edge("greet", END)

# グラフのコンパイル
app = graph.compile()

# 実行
result = app.invoke({"message": "はじめまして", "response": ""})
print(result["response"])
# 出力: こんにちは！はじめましてと言いましたね。
```

このコードでは、`State` という状態定義を行い、`greet_node` という処理を定義しています。
グラフは `START` から `greet` ノードを経由して `END` に到達します。

## 実装レベル2：ツール呼び出しを含むグラフ

実装レベル2では質問受付からツール実行から返答までの状態遷移を実装します。

### エージェントグラフの構築

ユーザーの質問に応じて、ツールを呼び出すエージェントを構築します。

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

class AgentState(TypedDict):
    messages: list

@tool
def search(query: str) -> str:
    """検索を実行します。"""
    # 実際にはAPIを呼び出すが、ここでは固定値を返す
    return f"検索結果: {query}についての情報が見つかりました。"

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools([search])

def agent_node(state: AgentState) -> AgentState:
    """LLMを呼び出してツール使用を判断するノード"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def tool_node(state: AgentState) -> AgentState:
    """ツールを実行するノード"""
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    results = []
    for tool_call in tool_calls:
        tool_result = search.invoke(tool_call["args"])
        results.append(ToolMessage(
            content=tool_result,
            tool_call_id=tool_call["id"]
        ))

    return {"messages": state["messages"] + results}

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """次のノードを決定する条件分岐"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

# グラフの構築
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

app = graph.compile()

# 実行
result = app.invoke({"messages": [HumanMessage(content="Pythonについて調べて")]})
for msg in result["messages"]:
    print(f"{msg.type}: {msg.content}")
```

このコードでは、`agent` ノードが LLM を呼び出し、ツールを使うべきか判断します。
ツールを使う場合は `tools` ノードに遷移し、そうでなければ終了します。
ツール実行後は再び `agent` ノードに戻り、結果を踏まえて最終回答を生成します。

## 運用観点：チェックポイントとヒューマンインザループ

運用観点としてチェックポイントとトレーシングとヒューマンインザループを扱います。

### チェックポイント（Checkpointing）

チェックポイントは、グラフの実行途中で状態を保存し、後から再開できる機能です。
長時間実行やヒューマンインザループに不可欠です。

```python
from langgraph.checkpoint.memory import MemorySaver

# チェックポイント機能を有効化
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# スレッドIDを指定して実行
config = {"configurable": {"thread_id": "conversation-1"}}
result = app.invoke({"messages": [HumanMessage(content="こんにちは")]}, config)

# 同じスレッドIDで続きを実行
result = app.invoke({"messages": result["messages"] + [HumanMessage(content="続きを教えて")]}, config)
```

チェックポイントを使うことで、会話履歴を保持し、コンテキストを維持できます。

### ヒューマンインザループ

ヒューマンインザループは、特定のノードで人間の承認を待つ機能です。
`interrupt_before` または `interrupt_after` を使ってノードの前後で中断できます。

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory, interrupt_before=["tools"])

config = {"configurable": {"thread_id": "conversation-1"}}
result = app.invoke({"messages": [HumanMessage(content="データを削除して")]}, config)

# ここでユーザーに確認を求める
print("ツールを実行しますか？ (yes/no)")
user_input = input()

if user_input == "yes":
    # 実行を再開
    result = app.invoke(None, config)
```

これにより、危険な操作を実行する前に人間の承認を得られます。

### トレーシング

LangSmith と統合することで、グラフの実行を可視化できます。
各ノードの実行時間、入出力、エラーをトレースできます。

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
```

LangSmith のダッシュボードでグラフの実行履歴を確認できます。

## 注意点と設計のトレードオフ

注意点として設計と運用コストと既存構成からの移行リスクを整理します。

### 設計の複雑性

LangGraph は柔軟性が高い反面、設計が複雑になりやすいです。
ノードとエッジを適切に分割し、状態を整理することが重要です。
小規模なタスクでは LangChain の高レベル API の方が適している場合があります。

### 運用コスト

チェックポイントを使う場合、状態を永続化するためのストレージが必要です。
長時間実行やヒューマンインザループを使う場合、インフラの設計が複雑になります。

### 既存構成からの移行

LangChain の `AgentExecutor` から LangGraph への移行は、コードの全面的な書き換えが必要です。
移行前に十分なテストを行い、段階的に移行することを推奨します。

## 本章のまとめ

本章では、LangGraph 1.0 の基本概念、グラフの構築方法、チェックポイントとヒューマンインザループの実装を解説しました。
LangGraph は複雑なワークフローを表現するための強力なツールですが、設計と運用コストを考慮する必要があります。
次章では、LangChain 1.0 の高レベル抽象を使った、より簡潔なエージェント構築方法を紹介します。
