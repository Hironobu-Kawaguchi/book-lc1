"""
LangChain Feature: content_blocks で推論過程を可視化

LangChain v1 の ChatModel は `response.content_blocks` で各種出力を
統一フォーマットで取得できます。

実行:
  uv run python ch05/feature_content_blocks.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv


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


def main() -> None:
    model = build_chat_model()

    question = "LangChain v1 の主要アップデートを3点でまとめて"
    response = model.invoke(question)

    content_blocks = getattr(response, "content_blocks", None)
    if not content_blocks:
        print("選択したモデルは content_blocks をまだサポートしていないようです。")
        print("OpenAI や Anthropic など対応プロバイダの API キーを設定してください。")
        return

    print("=== content_blocks ===")
    for block in content_blocks:
        block_type = block.get("type")
        if block_type == "reasoning":
            print(f"[reasoning] {block.get('reasoning')}")
        elif block_type == "text":
            print(f"[text] {block.get('text')}")
        elif block_type == "tool_call":
            print(f"[tool_call] {block.get('name')} => {block.get('args')}")
        else:
            print(f"[{block_type}] {block}")


if __name__ == "__main__":
    main()
