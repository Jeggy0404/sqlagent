"""Tools the agent can call. Every tool returns a string and raises ToolError on bad input."""
from __future__ import annotations

import ast
import json
import operator
import sqlite3
import time
from dataclasses import dataclass
from typing import Callable


class ToolError(Exception):
    """A recoverable error whose message is shown to the model so it can retry."""


# ---------- read-only SQL ----------

_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                    sqlite3.SQLITE_RECURSIVE}


_DENIED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit"}


def _authorizer(action, arg1, arg2, dbname, source):
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in _DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


class Database:
    """Read-only wrapper: opens the file in `mode=ro`, installs a deny-by-default
    authorizer, allows a single statement and caps the number of rows returned."""

    def __init__(self, path: str, max_rows: int = 50, timeout_s: float = 5.0):
        self.max_rows, self.timeout_s = max_rows, timeout_s
        uri = f"file:{path}?mode=ro" if path != ":memory:" else path
        self.conn = sqlite3.connect(uri, uri=path != ":memory:", check_same_thread=False)
        self.conn.set_authorizer(_authorizer)

    @classmethod
    def from_connection(cls, conn: sqlite3.Connection, max_rows: int = 50,
                        timeout_s: float = 5.0) -> "Database":
        self = cls.__new__(cls)
        self.max_rows, self.timeout_s, self.conn = max_rows, timeout_s, conn
        conn.set_authorizer(_authorizer)
        return self

    def tables(self) -> list[str]:
        # sqlite_master reads are allowed by the authorizer (SQLITE_READ)
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def describe(self, table: str) -> str:
        if table not in self.tables():
            raise ToolError(f"unknown table {table!r}; available: {', '.join(self.tables())}")
        sql = self.conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0]
        return sql

    def query(self, sql: str) -> str:
        stmt = sql.strip().rstrip(";").strip()
        if not stmt:
            raise ToolError("empty query")
        if ";" in stmt:
            raise ToolError("only one statement is allowed")
        if not stmt.lower().startswith(("select", "with")):
            raise ToolError("only SELECT queries are allowed")
        deadline = time.monotonic() + self.timeout_s
        self.conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
        try:
            cur = self.conn.execute(stmt)
            rows = cur.fetchmany(self.max_rows + 1)
        except sqlite3.Error as exc:
            msg = "query timed out" if "interrupt" in str(exc).lower() else f"SQL error: {exc}"
            raise ToolError(msg) from exc
        finally:
            self.conn.set_progress_handler(None, 0)
        cols = [d[0] for d in cur.description or []]
        truncated = len(rows) > self.max_rows
        rows = rows[: self.max_rows]
        out = {"columns": cols, "rows": [list(r) for r in rows], "row_count": len(rows)}
        if truncated:
            out["note"] = f"truncated to {self.max_rows} rows; add LIMIT or aggregate"
        return json.dumps(out, ensure_ascii=False, default=str)


# ---------- safe calculator ----------

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculate(expression: str) -> str:
    """Evaluate arithmetic only: numbers, + - * / // % **, parentheses. No names, no calls."""
    if len(expression) > 200:
        raise ToolError("expression too long")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ToolError("exponent too large")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
            return _UN[type(node.op)](ev(node.operand))
        raise ToolError("only plain arithmetic is allowed")

    try:
        result = ev(ast.parse(expression, mode="eval"))
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except SyntaxError as exc:
        raise ToolError("invalid expression") from exc
    return str(round(result, 10) if isinstance(result, float) else result)


# ---------- registry ----------

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def default_tools(db: Database) -> dict[str, Tool]:
    return {t.name: t for t in [
        Tool("list_tables", "List the tables and views in the database.", _obj({}, []),
             lambda: ", ".join(db.tables()) or "(no tables)"),
        Tool("describe_table", "Show the CREATE statement (columns and types) of a table.",
             _obj({"table": {"type": "string"}}, ["table"]), db.describe),
        Tool("run_sql", "Run one read-only SELECT query. Results are capped; prefer aggregates.",
             _obj({"sql": {"type": "string"}}, ["sql"]), db.query),
        Tool("calculate", "Evaluate an arithmetic expression exactly, e.g. '(120-95)/95*100'.",
             _obj({"expression": {"type": "string"}}, ["expression"]), calculate),
    ]}
