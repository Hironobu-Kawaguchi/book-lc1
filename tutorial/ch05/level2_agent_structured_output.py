"""
LangChain Level 2: `ToolStrategy` で構造化出力を直接取得

LangChain v1 では構造化出力がエージェントループに統合され、
追加のLLM呼び出しなしで `ToolStrategy` を介して Pydantic モデルを返せます。

実行:
  uv run python ch05/level2_agent_structured_output.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from pydantic import BaseModel, Field


def build_chat_model():
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


class RefundSummary(BaseModel):
    """返金可否を構造化して返すモデル。"""

    eligible: bool = Field(..., description="返金が可能かどうか")
    steps: str = Field(..., description="顧客への案内手順")


@tool
def lookup_order(order_id: str) -> str:
    """注文情報を返すダミーツール（実際はDB/APIを想定）。"""

    return (
        f"注文 {order_id} は発送済みで、未開封。\n"
        "返金ポリシー: 発送後30日以内、未使用なら全額返金可。"
    )


def main() -> None:
    llm = build_chat_model()

    agent = create_agent(
        model=llm,
        tools=[lookup_order],
        response_format=ToolStrategy(RefundSummary),
        system_prompt=(
            "あなたはカスタマーサクセス担当です。"
            "顧客の注文状況をツールで取得し、返金可否と案内手順を構造化して返してください。"
        ),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "注文 ORD-2025-001 の返金可否を教えて",
                }
            ]
        }
    )

    structured = result.get("structured_response") or result.get("structured_output")
    print("=== 構造化レスポンス ===")
    print(structured)


if __name__ == "__main__":
    main()
