#!/usr/bin/env python3
"""
Agent Tool Calling Demo
=======================

Multi-turn conversation with Llama Stack's responses API (0.7.x+).
The agent uses the search_documents tool to answer questions with RAG context.

    python agent_tool_calling_client.py --query "How does SSH work on RHEL?"

"""

import argparse

from llama_stack_client import LlamaStackClient


SEARCH_TOOL = {
    "type": "function",
    "name": "search_documents",
    "description": (
        "Search RHEL knowledge base using advanced RAG techniques. "
        "Returns relevant document chunks with similarity scores."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query or question",
            },
            "max_chunks": {
                "type": "integer",
                "description": "Maximum number of document chunks to retrieve",
            },
            "similarity_threshold": {
                "type": "number",
                "description": "Minimum similarity threshold (0.0-1.0)",
            },
        },
        "required": ["query"],
    },
}

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant with access to a RHEL knowledge base. "
    "When asked a question, use the search_documents tool to find "
    "relevant information, then provide a comprehensive answer based "
    "on the search results."
)


def run_demo(base_url, query):
    client = LlamaStackClient(base_url=base_url)

    try:
        client.models.list()
    except Exception:
        print(f"Cannot connect to Llama Stack server at {base_url}")
        return False

    print(f"Query: {query}")
    print("-" * 50)

    response = client.responses.create(
        model="ollama/qwen2.5:7b-instruct",
        instructions=SYSTEM_INSTRUCTIONS,
        input=query,
        tools=[SEARCH_TOOL],
        tool_choice="auto",
        stream=True,
    )

    tool_calls = []
    for chunk in response:
        if hasattr(chunk, "type"):
            if chunk.type == "response.output_item.added":
                item = chunk.item
                if hasattr(item, "type") and item.type == "function_call":
                    tool_calls.append(item)
                    print(f"\n[Tool Call] {item.name}({item.arguments})")
            elif chunk.type == "response.output_text.delta":
                print(chunk.delta, end="", flush=True)
            elif chunk.type == "response.completed":
                break

    print()
    print("-" * 50)

    if tool_calls:
        print(f"\nTool calls made: {len(tool_calls)}")
        for tc in tool_calls:
            print(f"  - {tc.name}: {tc.arguments}")

    print("\nCheck server logs to verify RAG tool execution")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Agent Tool Calling Demo - Llama Stack responses API"
    )
    parser.add_argument(
        "--query",
        default="How does solar energy compare to other renewable energy sources?",
        help="Query to ask the agent",
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8321",
        help="Llama Stack server URL",
    )

    args = parser.parse_args()
    run_demo(args.server, args.query)


if __name__ == "__main__":
    main()
