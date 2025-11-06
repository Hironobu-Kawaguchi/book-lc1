"""
LangChain Level 1: `create_agent` で FAQ エージェントを最短構築

LangChain v1 では `create_agent` が安定APIとなり、モデル・ツール・プロンプトを
まとめて渡すだけでエージェントを生成できます。

実行:
  cp tutorial/.env.sample tutorial/.env  # 初回のみ
  # .env に API キー（OPENAI_API_KEY または ANTHROPIC_API_KEY）を設定
  uv run python ch05/level1_agent_faq.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool


def build_chat_model():
    """環境変数に応じて OpenAI か Anthropic のチャットモデルを返す。"""

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


@tool
def faq_knowledge() -> str:
    """社内ナレッジに固定で登録されている返金ルールを返す。"""

    return (
        "返金は購入後30日以内にサポートフォームから申請してください。\n"
        "未使用品は全額返金。\n"
        "使用済みの場合は個別審査で一部返金となります。"
    )


def main() -> None:
    llm = build_chat_model()

    # LangChain v1 の create_agent でエージェントを生成
    agent = create_agent(
        model=llm,
        tools=[faq_knowledge],
        system_prompt=(
            "あなたはカスタマーサポート担当です。"
            "ユーザの質問に対し、社内ナレッジツール（faq_knowledge）を必要に応じて呼び出し"
            "簡潔なFAQ形式で回答してください。"
        ),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "返金ポリシーのポイントを3行で教えてください。",
                }
            ]
        }
    )

    # create_agent は最終メッセージを messages 配列の末尾に格納する
    messages = result["messages"]
    final_message = messages[-1]
    if hasattr(final_message, "content"):
        content = final_message.content
    elif isinstance(final_message, dict):
        content = final_message.get("content", "")
    else:
        content = str(final_message)
    print(content)


if __name__ == "__main__":
    main()
