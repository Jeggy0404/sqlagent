"""LLM interface and clients.

An LLM is any object with `chat(messages, tools) -> Message`. Messages use the
OpenAI chat format (dicts), so the same history works with any compatible API.
ScriptedLLM replays a fixed list of replies and is what the tests use.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model produced it


@dataclass
class Message:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> Message: ...


class ScriptedLLM:
    """Replays prepared replies; records what it was asked. For tests and demos."""

    def __init__(self, replies: list[Message]):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict]) -> Message:
        self.calls.append([dict(m) for m in messages])
        if not self.replies:
            raise RuntimeError("ScriptedLLM ran out of replies")
        return self.replies.pop(0)


class OpenAIChat:
    """OpenAI-compatible chat completions client with tool calling (stdlib only)."""

    def __init__(self, model: str = "gpt-4o-mini", base_url: str = "https://api.openai.com/v1",
                 api_key: str | None = None):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def chat(self, messages: list[dict], tools: list[dict]) -> Message:
        body = {"model": self.model, "temperature": 0, "messages": messages, "tools": tools}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            msg = json.load(resp)["choices"][0]["message"]
        calls = [ToolCall(c["id"], c["function"]["name"], c["function"]["arguments"])
                 for c in msg.get("tool_calls") or []]
        return Message(content=msg.get("content") or "", tool_calls=calls)
