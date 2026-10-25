# sqlagent

A small tool-calling agent that answers natural-language questions about a SQLite database. The interesting part is not the prompt but the **guardrails around the loop**: the agent can read data, and nothing else.

This is a personal portfolio project with no users. It has zero runtime dependencies (standard library only).

## How it works

```
question -> LLM -> tool calls (list_tables, describe_table, run_sql, calculate) -> results -> LLM -> ... -> answer
```

| Module | Responsibility |
| --- | --- |
| `sqlagent/agent.py` | the loop, argument validation, limits, step trace |
| `sqlagent/tools.py` | read-only SQL, safe calculator, tool registry |
| `sqlagent/llm.py` | LLM interface, `OpenAIChat` (OpenAI-compatible), `ScriptedLLM` for tests |
| `sqlagent/demo_db.py` | deterministic demo database (customers, products, orders) |
| `sqlagent/eval.py` | accuracy benchmark with ground truth computed by SQL |

## Guardrails (all covered by tests)

- **Read-only SQL, enforced by the database, not by string checks.** SQLite opens in `mode=ro` and a deny-by-default authorizer allows only SELECT/READ/FUNCTION, so `INSERT`, `DELETE`, `DROP`, `PRAGMA`, `ATTACH` and writes hidden inside a CTE all fail. A prefix check (`SELECT`/`WITH`) and a single-statement rule are additional layers, not the only one.
- **Query timeout** (default 5 s) via a progress handler, so a runaway recursive CTE is interrupted.
- **Row cap** (default 50) with a note telling the model to aggregate.
- **Calculator parses an AST** and allows only numbers and arithmetic operators. No names, calls, attributes or huge exponents.
- **Strict tool arguments:** invalid JSON, unknown tools, missing or extra arguments become error messages the model can react to, never exceptions.
- **Loop limits:** maximum steps, an identical repeated tool call stops the run, and consecutive tool errors stop the run. Every run returns a trace of steps and a `stop_reason`.
- **Result truncation** keeps huge tool outputs out of the context window.

## Usage

```bash
export OPENAI_API_KEY=...
python -m sqlagent.cli "Which product category generated the most revenue?"
```

```python
from sqlagent import Agent
from sqlagent.llm import OpenAIChat
from sqlagent.tools import Database

agent = Agent(OpenAIChat(), Database("my.db"))   # file is opened read-only
result = agent.run("How many orders were placed per country?")
print(result.answer, result.stop_reason)
for step in result.steps: print(step.tool, step.arguments, step.ok)
```

Any object with `chat(messages, tools) -> Message` works as the LLM, so other providers only need a small adapter.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The 27 tests use `ScriptedLLM` (replays prepared model replies), so they are deterministic and need no network. **They verify the loop and the safety properties. They do not measure how good any model is at the task.**

## Accuracy benchmark

`python -m sqlagent.eval --model <name>` runs 8 questions whose correct numbers are computed directly from the database, then checks that the answer contains the right value. I have not published scores here: the harness is tested, but results depend on the model and I have not run it against a live model for this repository. Run it with your key and put your own numbers in this section.

## Limits

- Only SQLite. The tool layer is small, so another database needs its own read-only enforcement (use a read-only role, not string filtering).
- A correct-looking answer is not proof of correctness: the model can still write a wrong but valid query. The trace of executed SQL is returned so answers can be audited.
- The benchmark is small and uses a synthetic database.

## License

MIT
