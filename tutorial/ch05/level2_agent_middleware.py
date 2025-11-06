"""
LangChain Level 2: Middleware で入力・出力を前処理／後処理

LangChain v1 では `create_agent` に `middleware` 引数が追加され、
エージェントに渡るメッセージをフックしてPIIマスク等を施せます。

実行:
  uv run python ch05/level2_agent_middleware.py
"""

from __future__ import annotations

import os
import re
from typing import Sequence

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import tool


def build_chat_model():
    """環境変数のAPIキーに合わせてLLMクライアントを返す。"""

    load_dotenv()
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.0)
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
            temperature=0.0,
        )
    raise EnvironmentError(
        "OPENAI_API_KEY か ANTHROPIC_API_KEY のどちらかを .env に設定してください。"
    )


class PIIMiddleware(AgentMiddleware):
    """電話番号を簡易マスクするミドルウェア。"""

    def __init__(self, patterns: Sequence[str]) -> None:
        self._compiled = [re.compile(p) for p in patterns]

    def before_agent(self, state: AgentState, runtime) -> AgentState:  # type: ignore[override]
        message: HumanMessage = state["messages"][-1]
        masked_content = message.content
        for pattern in self._compiled:
            masked_content = pattern.sub("[masked]", masked_content)
        new_messages = [*state["messages"][:-1], HumanMessage(content=masked_content)]
        return {**state, "messages": new_messages}  # type: ignore[return-value]

    def after_agent(self, state: AgentState, runtime) -> AgentState:  # type: ignore[override]
        # 応答側にも同様の簡易マスクを施す（messages の末尾が AIMessage を想定）
        messages = state.get("messages", [])
        if not messages:
            return state
        last = messages[-1]
        if isinstance(last, AIMessage):
            masked_content = last.content
            for pattern in self._compiled:
                masked_content = pattern.sub("[masked]", masked_content)
            messages = [*messages[:-1], AIMessage(content=masked_content)]
            return {**state, "messages": messages}  # type: ignore[return-value]
        return state


@tool
def lookup_customer_policy(customer_id: str) -> str:
    """顧客IDから社内ポリシーの概要を取得するダミーツール。"""

    return (
        f"顧客 {customer_id} の返金ポリシーは標準プランです。\n"
        "返品受付は30日以内、出荷元への返送が必要です。"
    )


def main() -> None:
    llm = build_chat_model()

    agent = create_agent(
        model=llm,
        tools=[lookup_customer_policy],
        middleware=[PIIMiddleware(patterns=[r"\b\d{4}-\d{4}-\d{4}\b"])],
    )

    user_question = "顧客 1234-5678-9999 の返金ポリシーを要約して"
    result = agent.invoke({"messages": [{"role": "user", "content": user_question}]})

    final_message = result["messages"][-1]
    if hasattr(final_message, "content"):
        content = final_message.content
    elif isinstance(final_message, dict):
        content = final_message.get("content", "")
    else:
        content = str(final_message)
    print("=== エージェント応答 ===")
    print(content)


if __name__ == "__main__":
    main()
