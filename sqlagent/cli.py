"""python -m sqlagent.cli "question"  (uses the demo DB, needs OPENAI_API_KEY)"""
from __future__ import annotations

import sys

from .agent import Agent
from .demo_db import build
from .llm import OpenAIChat
from .tools import Database


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('usage: python -m sqlagent.cli "your question"')
    agent = Agent(OpenAIChat(), Database.from_connection(build()))
    res = agent.run(" ".join(sys.argv[1:]))
    for s in res.steps:
        print(f"[{s.n}] {s.tool}({s.arguments}) -> {'ok' if s.ok else 'ERR'}")
    print(f"\n{res.answer}\n(stop: {res.stop_reason})")


if __name__ == "__main__":
    main()
