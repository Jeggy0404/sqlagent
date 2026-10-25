"""The agent loop: model -> tool calls -> results -> model, with hard limits."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import LLM, Message
from .tools import Database, Tool, ToolError, default_tools

SYSTEM = (
    "You answer questions about a SQLite database. Always inspect the schema before writing SQL. "
    "Use run_sql for data and calculate for arithmetic; never guess numbers. "
    "If the data cannot answer the question, say so. Give the final answer as plain text."
)


@dataclass
class Step:
    n: int
    tool: str
    arguments: str
    result: str
    ok: bool


@dataclass
class AgentResult:
    answer: str
    finished: bool            # False if a limit stopped the run
    stop_reason: str          # "answer" | "max_steps" | "repeated_call" | "too_many_errors"
    steps: list[Step] = field(default_factory=list)


class Agent:
    def __init__(self, llm: LLM, db: Database, max_steps: int = 8, max_errors: int = 3,
                 max_result_chars: int = 4000, tools: dict[str, Tool] | None = None):
        self.llm, self.db = llm, db
        self.max_steps, self.max_errors, self.max_result_chars = max_steps, max_errors, max_result_chars
        self.tools = tools or default_tools(db)

    def _run_tool(self, name: str, raw_args: str) -> tuple[str, bool]:
        tool = self.tools.get(name)
        if tool is None:
            return f"error: unknown tool {name!r}; available: {', '.join(self.tools)}", False
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            return "error: arguments are not valid JSON", False
        if not isinstance(args, dict):
            return "error: arguments must be a JSON object", False
        props = tool.parameters.get("properties", {})
        missing = [k for k in tool.parameters.get("required", []) if k not in args]
        extra = [k for k in args if k not in props]
        if missing or extra:
            return f"error: missing {missing or 'none'}, unexpected {extra or 'none'}", False
        try:
            out = tool.fn(**args)
        except ToolError as exc:
            return f"error: {exc}", False
        except TypeError as exc:
            return f"error: bad arguments: {exc}", False
        if len(out) > self.max_result_chars:
            out = out[: self.max_result_chars] + "\n...[truncated]"
        return out, True

    def run(self, question: str) -> AgentResult:
        messages: list[dict] = [{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": question}]
        schemas = [t.schema() for t in self.tools.values()]
        steps: list[Step] = []
        seen: set[tuple[str, str]] = set()
        errors = 0
        for _ in range(self.max_steps):
            reply: Message = self.llm.chat(messages, schemas)
            if not reply.tool_calls:
                return AgentResult(reply.content.strip(), True, "answer", steps)
            messages.append({"role": "assistant", "content": reply.content or None, "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in reply.tool_calls]})
            for call in reply.tool_calls:
                key = (call.name, call.arguments)
                if key in seen:
                    return AgentResult("Stopped: the model repeated an identical tool call.",
                                       False, "repeated_call", steps)
                seen.add(key)
                result, ok = self._run_tool(call.name, call.arguments)
                steps.append(Step(len(steps) + 1, call.name, call.arguments, result, ok))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                errors = 0 if ok else errors + 1
                if errors >= self.max_errors:
                    return AgentResult("Stopped: too many consecutive tool errors.",
                                       False, "too_many_errors", steps)
        return AgentResult("Stopped: step limit reached without a final answer.",
                           False, "max_steps", steps)
