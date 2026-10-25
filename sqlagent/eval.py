"""Answer-accuracy benchmark against ground truth computed directly from the demo DB.

    OPENAI_API_KEY=... python -m sqlagent.eval [--model gpt-4o-mini]

Each task's expected value comes from a trusted SQL query, and an answer passes if it
contains that number (thousands separators and rounding to 2 decimals tolerated).
The checker is unit-tested; scores depend on the model you plug in.
"""
from __future__ import annotations

import argparse
import re

from .agent import Agent
from .demo_db import build
from .llm import OpenAIChat
from .tools import Database

TASKS = [
    ("How many orders are there in total?", "SELECT COUNT(*) FROM orders"),
    ("How many customers are from Germany (DE)?", "SELECT COUNT(*) FROM customers WHERE country='DE'"),
    ("What is the total quantity of items ordered?", "SELECT SUM(quantity) FROM orders"),
    ("What is the price of the most expensive product?", "SELECT MAX(price) FROM products"),
    ("How many distinct customers placed at least one order?", "SELECT COUNT(DISTINCT customer_id) FROM orders"),
    ("How many orders were placed in March 2025?", "SELECT COUNT(*) FROM orders WHERE ordered_on LIKE '2025-03-%'"),
    ("What is the total revenue (price times quantity) across all orders?",
     "SELECT SUM(p.price*o.quantity) FROM orders o JOIN products p ON p.id=o.product_id"),
    ("How many products are in the 'tools' category?", "SELECT COUNT(*) FROM products WHERE category='tools'"),
]


def numbers_in(text: str) -> set[float]:
    cleaned = re.sub(r"(?<=\d)[,\s](?=\d{3}\b)", "", text)
    return {round(float(m), 2) for m in re.findall(r"-?\d+(?:\.\d+)?", cleaned)}


def is_correct(answer: str, expected: float) -> bool:
    return round(float(expected), 2) in numbers_in(answer)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()
    conn = build()
    agent = Agent(OpenAIChat(model=args.model), Database.from_connection(conn))
    passed = 0
    for question, truth_sql in TASKS:
        expected = conn.execute(truth_sql).fetchone()[0]
        res = agent.run(question)
        ok = res.finished and is_correct(res.answer, expected)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  steps={len(res.steps)}  {question}  (expected {expected})")
    print(f"\n{passed}/{len(TASKS)} correct with {args.model}")


if __name__ == "__main__":
    main()
