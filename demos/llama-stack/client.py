#!/usr/bin/env python3
"""
Docs2DB RAG Demo Client
=======================

Uses the Llama Stack responses API to invoke the search_documents tool
via the LLM (0.7.x+).

    python client.py --query "What is Solar Power?"

"""

import argparse
import sys

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


def run_query(base_url, query, max_chunks=5, similarity_threshold=0.7):
    client = LlamaStackClient(base_url=base_url)

    try:
        client.models.list()
    except Exception:
        print(f"Cannot connect to Llama Stack server at {base_url}")
        return False

    print(f"Query: {query}")
    print(f"Parameters: max_chunks={max_chunks}, similarity_threshold={similarity_threshold}")
    print()

    response = client.responses.create(
        model="ollama/qwen2.5:7b-instruct",
        input=f"Use the search_documents tool to answer: {query}",
        tools=[SEARCH_TOOL],
        tool_choice="required",
        stream=False,
    )

    for item in response.output:
        if item.type == "function_call":
            print(f"Tool called: {item.name}")
            print(f"Arguments: {item.arguments}")
            print()
        elif item.type == "message":
            content = item.content
            if isinstance(content, list):
                for part in content:
                    if hasattr(part, "text"):
                        print(part.text)
            elif isinstance(content, str):
                print(content)

    return True


def main():
    parser = argparse.ArgumentParser(description="Docs2DB RAG Search Demo")
    parser.add_argument(
        "--query",
        default="How do I configure SSH key-based authentication?",
        help="Query to search for",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=5,
        help="Maximum number of chunks to return (default: 5)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for search (default: 0.7)",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8321",
        help="Llama Stack server URL (default: http://localhost:8321)",
    )

    args = parser.parse_args()
    success = run_query(
        args.server_url,
        args.query,
        max_chunks=args.max_chunks,
        similarity_threshold=args.similarity_threshold,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
