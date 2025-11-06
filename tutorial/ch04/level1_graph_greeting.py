"""
LangGraph Level 1: 最小グラフ（挨拶→返答）

実行:
  uv run python ch04/level1_graph_greeting.py
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph


class GreetState(TypedDict):
    """グラフ内で共有する状態。入力と出力を1つの辞書で扱う。"""

    name: str
    reply: str


def greet(state: GreetState) -> dict:
    """挨拶メッセージを組み立てるノード。"""

    name = state.get("name", "World")
    return {"reply": f"Hello, {name}!"}


def main() -> None:
    # StateGraphでノード（処理ステップ）とエッジ（遷移）を定義する
    graph = StateGraph(GreetState)
    graph.add_node("greet", greet)  # 単一ノードのグラフ
    graph.set_entry_point("greet")  # 実行開始点を指定
    graph.add_edge("greet", END)  # ノードの後は終了
    app = graph.compile()

    # グラフに初期状態を渡して実行
    result = app.invoke({"name": "LangGraph"})
    print(result["reply"])  # -> Hello, LangGraph!


if __name__ == "__main__":
    main()
