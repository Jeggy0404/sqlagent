import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlagent import Agent, Message, ToolCall
from sqlagent.demo_db import build
from sqlagent.llm import ScriptedLLM
from sqlagent.tools import Database, ToolError, calculate


def call(name, **args):
    return Message(tool_calls=[ToolCall(f"c{abs(hash((name, json.dumps(args, sort_keys=True))))}", name, json.dumps(args))])


def final(text):
    return Message(content=text)


def make_db(**kw):
    return Database.from_connection(build(), **kw)


class SqlSafetyTests(unittest.TestCase):
    def setUp(self):
        self.db = make_db()

    def test_select_works(self):
        out = json.loads(self.db.query("SELECT COUNT(*) AS n FROM orders"))
        self.assertEqual(out["columns"], ["n"])
        self.assertEqual(out["rows"], [[300]])

    def test_with_cte_works(self):
        out = json.loads(self.db.query("WITH t AS (SELECT 1 AS x) SELECT x FROM t"))
        self.assertEqual(out["rows"], [[1]])

    def test_writes_are_rejected(self):
        for sql in ["INSERT INTO customers VALUES(999,'x','CZ')", "DELETE FROM orders",
                    "UPDATE products SET price=0", "DROP TABLE orders",
                    "CREATE TABLE evil(x)", "PRAGMA writable_schema=1"]:
            with self.assertRaises(ToolError, msg=sql):
                self.db.query(sql)
        self.assertEqual(json.loads(self.db.query("SELECT COUNT(*) FROM orders"))["rows"], [[300]])

    def test_write_hidden_inside_cte_is_denied_by_authorizer(self):
        with self.assertRaises(ToolError):
            self.db.query("WITH x AS (SELECT 1) DELETE FROM orders")
        self.assertEqual(json.loads(self.db.query("SELECT COUNT(*) FROM orders"))["rows"], [[300]])

    def test_multiple_statements_rejected(self):
        with self.assertRaises(ToolError):
            self.db.query("SELECT 1; DROP TABLE orders")

    def test_attach_and_extensions_denied(self):
        for sql in ["SELECT load_extension('evil')", "ATTACH DATABASE ':memory:' AS x",
                    "SELECT * FROM pragma_table_info('orders') WHERE name='x' AND 1=1; PRAGMA foreign_keys=0"]:
            with self.assertRaises(ToolError, msg=sql):
                self.db.query(sql)

    def test_row_cap_and_truncation_note(self):
        out = json.loads(make_db(max_rows=5).query("SELECT * FROM orders"))
        self.assertEqual(out["row_count"], 5)
        self.assertIn("truncated", out["note"])

    def test_runaway_query_times_out(self):
        db = make_db(timeout_s=0.3)
        with self.assertRaises(ToolError) as ctx:
            db.query("WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r) SELECT COUNT(*) FROM r")
        self.assertIn("timed out", str(ctx.exception))

    def test_file_database_is_opened_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.db")
            c = build(path)
            c.close()
            db = Database(path)
            self.assertEqual(json.loads(db.query("SELECT COUNT(*) FROM customers"))["rows"], [[40]])
            with self.assertRaises(ToolError):
                db.query("DELETE FROM customers")
            db.conn.close()

    def test_describe_unknown_table(self):
        with self.assertRaises(ToolError):
            self.db.describe("nope; DROP TABLE orders")

    def test_sql_error_is_a_recoverable_tool_error(self):
        with self.assertRaises(ToolError) as ctx:
            self.db.query("SELECT missing_col FROM orders")
        self.assertIn("SQL error", str(ctx.exception))


class CalculatorTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(calculate("(120-95)/95*100"), "26.3157894737")
        self.assertEqual(calculate("2**10 + 7 % 4"), "1027")

    def test_rejects_code(self):
        for bad in ["__import__('os').system('ls')", "open('/etc/passwd')", "1 if True else 2",
                    "[1,2]", "abs(-1)", "x + 1", "True + 1", "'a'*3"]:
            with self.assertRaises(ToolError, msg=bad):
                calculate(bad)

    def test_limits(self):
        with self.assertRaises(ToolError):
            calculate("9**9**9")
        with self.assertRaises(ToolError):
            calculate("1/0")


class AgentLoopTests(unittest.TestCase):
    def test_happy_path(self):
        llm = ScriptedLLM([call("list_tables"), call("run_sql", sql="SELECT COUNT(*) FROM orders"),
                           final("There are 300 orders.")])
        res = Agent(llm, make_db()).run("How many orders?")
        self.assertTrue(res.finished)
        self.assertEqual(res.stop_reason, "answer")
        self.assertEqual([s.tool for s in res.steps], ["list_tables", "run_sql"])
        self.assertIn("300", res.steps[1].result)

    def test_tool_results_are_sent_back_to_the_model(self):
        llm = ScriptedLLM([call("list_tables"), final("ok")])
        Agent(llm, make_db()).run("q")
        last = llm.calls[-1]
        self.assertEqual(last[-1]["role"], "tool")
        self.assertIn("customers", last[-1]["content"])
        self.assertEqual(last[-2]["role"], "assistant")

    def test_model_can_recover_from_a_sql_error(self):
        llm = ScriptedLLM([call("run_sql", sql="SELECT nope FROM orders"),
                           call("run_sql", sql="SELECT COUNT(*) FROM orders"), final("300")])
        res = Agent(llm, make_db()).run("count")
        self.assertEqual([s.ok for s in res.steps], [False, True])
        self.assertTrue(res.finished)

    def test_unknown_tool_and_bad_json_are_reported_not_raised(self):
        bad_json = Message(tool_calls=[ToolCall("a", "run_sql", "{not json")])
        llm = ScriptedLLM([call("drop_everything"), bad_json, final("done")])
        res = Agent(llm, make_db()).run("q")
        self.assertIn("unknown tool", res.steps[0].result)
        self.assertIn("not valid JSON", res.steps[1].result)

    def test_extra_or_missing_arguments_rejected(self):
        llm = ScriptedLLM([call("run_sql", sql="SELECT 1", mode="write"), call("run_sql"), final("x")])
        res = Agent(llm, make_db()).run("q")
        self.assertFalse(res.steps[0].ok)
        self.assertFalse(res.steps[1].ok)

    def test_attack_through_the_agent_is_blocked_and_data_survives(self):
        db = make_db()
        llm = ScriptedLLM([call("run_sql", sql="DELETE FROM orders"), final("could not")])
        res = Agent(llm, db).run("delete all orders")
        self.assertFalse(res.steps[0].ok)
        self.assertEqual(json.loads(db.query("SELECT COUNT(*) FROM orders"))["rows"], [[300]])

    def test_step_limit(self):
        llm = ScriptedLLM([call("run_sql", sql=f"SELECT {i}") for i in range(20)])
        res = Agent(llm, make_db(), max_steps=4).run("loop")
        self.assertFalse(res.finished)
        self.assertEqual(res.stop_reason, "max_steps")
        self.assertEqual(len(res.steps), 4)

    def test_identical_repeated_call_stops_the_run(self):
        llm = ScriptedLLM([call("list_tables"), call("list_tables"), final("never reached")])
        res = Agent(llm, make_db()).run("q")
        self.assertEqual(res.stop_reason, "repeated_call")
        self.assertEqual(len(res.steps), 1)

    def test_consecutive_errors_stop_the_run(self):
        llm = ScriptedLLM([call("run_sql", sql=f"SELECT bad{i} FROM orders") for i in range(5)])
        res = Agent(llm, make_db(), max_errors=3).run("q")
        self.assertEqual(res.stop_reason, "too_many_errors")
        self.assertEqual(len(res.steps), 3)

    def test_large_results_are_truncated_for_the_model(self):
        llm = ScriptedLLM([call("run_sql", sql="SELECT * FROM orders"), final("x")])
        res = Agent(llm, make_db(max_rows=200), max_result_chars=300).run("q")
        self.assertLessEqual(len(res.steps[0].result), 300 + 20)
        self.assertIn("[truncated]", res.steps[0].result)

    def test_calculator_tool_through_agent(self):
        llm = ScriptedLLM([call("calculate", expression="(120-95)/95*100"), final("26.3%")])
        res = Agent(llm, make_db()).run("pct change")
        self.assertEqual(res.steps[0].result, "26.3157894737")



class EvalCheckerTests(unittest.TestCase):
    def test_number_matching(self):
        from sqlagent.eval import TASKS, is_correct
        self.assertTrue(is_correct("There are 1,234 orders.", 1234))
        self.assertTrue(is_correct("Revenue is 12345.678 EUR", 12345.68))
        self.assertFalse(is_correct("There are 300 orders.", 301))
        self.assertFalse(is_correct("I could not tell.", 5))

    def test_every_ground_truth_query_runs(self):
        from sqlagent.eval import TASKS
        conn = build()
        for question, sql in TASKS:
            self.assertIsNotNone(conn.execute(sql).fetchone()[0], question)


if __name__ == "__main__":
    unittest.main()
