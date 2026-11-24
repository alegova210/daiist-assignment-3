"""
Assignment 3: the agent harness (`main.py ask "question"` runs `python agent.py "question"`).

LangChain `create_agent` on Gemini. Tools come from server.py over MCP (stdio). AGENTS.md is the
system prompt.

Guardrails
  1. Human approval before add_document runs (asked in the terminal; denied when nobody can answer,
     e.g. during evaluation).
  2. Call limits: at most MAX_SEARCHES searches per question (the tool answers "limit reached"
     after that) and at most MAX_MODEL_CALLS model calls.
  3. Citation guardrail (our choice): every citation in the answer must point to a passage that was
     actually retrieved in this run. An answer that is not a refusal and has no valid citation is
     blocked and replaced by a refusal, because in this use case an unsupported answer is the most
     expensive failure.

`ask(question)` returns the answer plus the trace (tool calls, retrieved passages, tokens, seconds)
for evaluate.py.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from pathlib import Path

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODELS = ["gemini-3.7-flash", "gemini-3.5-flash"]   # used when MODEL is overloaded (503)
MAX_SEARCHES = 4
MAX_MODEL_CALLS = 8
REFUSAL = "El corpus no cubre esta pregunta."
CITATION_RE = re.compile(r"\[([A-Z][A-Z0-9_]+),\s*([^\]\(]+?)\s*(?:\(([^)]*)\))?\]")
RETRIEVED_RE = re.compile(r"Cita: \[([^,\]]+), ([^\]\(]+?)\s*(?:\(([^)]*)\))?\]")


def system_prompt() -> str:
    return (ROOT / "AGENTS.md").read_text(encoding="utf-8")


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def check_citations(answer: str, retrieved: set[tuple[str, str]]) -> tuple[list, list]:
    valid, invalid = [], []
    for source, location, _sub in CITATION_RE.findall(answer):
        (valid if (source, location.strip()) in retrieved else invalid).append(f"{source}, {location.strip()}")
    return valid, invalid


async def _ask(question: str, interactive: bool) -> dict:
    from langchain.agents import create_agent
    from langchain_core.tools import StructuredTool
    from langchain_google_genai import ChatGoogleGenerativeAI
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    load_api_key()
    started = time.time()
    trace: list[dict] = []
    retrieved: set[tuple[str, str]] = set()
    passages: list[str] = []
    searches = 0

    params = StdioServerParameters(command=sys.executable, args=["server.py"], cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools

            def wrap(tool):
                async def call(**kwargs):
                    nonlocal searches
                    if tool.name == "search":
                        searches += 1
                        if searches > MAX_SEARCHES:
                            trace.append({"tool": tool.name, "args": kwargs, "result": "LIMIT"})
                            return f"Search limit ({MAX_SEARCHES}) reached. Answer with the passages you have."
                    if tool.name == "add_document":
                        approved = False
                        if interactive and sys.stdin.isatty():
                            reply = input(f"\nThe agent wants to run add_document({kwargs}). Approve? [y/N] ")
                            approved = reply.strip().lower() in {"y", "yes", "s", "si", "sí"}
                        if not approved:
                            trace.append({"tool": tool.name, "args": kwargs, "result": "DENIED"})
                            return "The user did not approve add_document. Do not retry it."
                    result = await session.call_tool(tool.name, kwargs)
                    text = "\n".join(getattr(c, "text", "") for c in result.content)
                    trace.append({"tool": tool.name, "args": kwargs, "result": text})
                    if tool.name == "search":
                        for source, location, _ in RETRIEVED_RE.findall(text):
                            retrieved.add((source, location.strip()))
                        passages.append(text)
                    return text
                return StructuredTool.from_function(coroutine=call, name=tool.name,
                                                    description=tool.description or "",
                                                    args_schema=getattr(tool, "input_schema", None) or getattr(tool, "inputSchema"))

            tools = [wrap(t) for t in mcp_tools]
            model = ChatGoogleGenerativeAI(model=MODEL, temperature=0, max_retries=3)
            middleware = []
            try:
                from langchain.agents.middleware import ModelFallbackMiddleware
                middleware.append(ModelFallbackMiddleware(
                    *[ChatGoogleGenerativeAI(model=m, temperature=0, max_retries=2) for m in FALLBACK_MODELS]))
            except ImportError:
                pass
            try:
                from langchain.agents.middleware import ModelCallLimitMiddleware
                middleware.append(ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"))
            except ImportError:
                pass
            agent = create_agent(model, tools, system_prompt=system_prompt(), middleware=middleware)
            state = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})

    messages = state["messages"]
    answer = _text(messages[-1].content).strip()
    tokens_in = tokens_out = 0
    for m in messages:
        usage = getattr(m, "usage_metadata", None) or {}
        tokens_in += usage.get("input_tokens", 0)
        tokens_out += usage.get("output_tokens", 0)

    refused = answer.startswith(REFUSAL)
    valid, invalid = check_citations(answer, retrieved)
    guardrail = None
    if not refused and not valid:
        guardrail = "blocked: answer without any valid citation"
        answer = (f"{REFUSAL} (La respuesta generada no tenía citas verificables en el corpus, "
                  f"así que no la muestro.)")
        refused = True
    return {
        "question": question, "answer": answer, "refused": refused,
        "valid_citations": valid, "invalid_citations": invalid, "guardrail": guardrail,
        "searches": searches, "trace": trace, "passages": "\n\n".join(passages),
        "tokens_in": tokens_in, "tokens_out": tokens_out,
        "seconds": round(time.time() - started, 1), "model": MODEL,
    }


def ask(question: str, interactive: bool = False) -> dict:
    return asyncio.run(_ask(question, interactive))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('usage: python agent.py "your question"')
    result = ask(" ".join(sys.argv[1:]), interactive=True)
    print(result["answer"])
    if result["valid_citations"]:
        print("\nFuentes:")
        for c in dict.fromkeys(result["valid_citations"]):
            print(f"  - {c}")
    if result["invalid_citations"]:
        print(f"\n(Citas no verificadas: {', '.join(result['invalid_citations'])})")
    print(f"\n[{result['searches']} búsquedas, {result['tokens_in']}+{result['tokens_out']} tokens, "
          f"{result['seconds']} s, {result['model']}]")


if __name__ == "__main__":
    main()
