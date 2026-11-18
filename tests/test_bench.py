import json
import unittest

from agentbench.agents import AGENTS
from agentbench.env import Env, tool_calculator, tool_convert, tool_lookup
from agentbench.llm import LLMResponse, ModelUnavailable
from agentbench.runner import run_one, summarise
from agentbench.tasks import BY_ID, TASKS


def call(name, args, i=0):
    return {"id": f"c{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class FakeClient:
    """Plays back scripted replies. `bad` models raise, to test fallback."""
    base_url = "fake"

    def __init__(self, script, bad=("unavailable/model-does-not-exist",)):
        self.script, self.bad, self.calls = list(script), set(bad), []

    def chat(self, models, messages, tools=None, temperature=0):
        errs = []
        for m in models:
            if m in self.bad:
                errs.append(f"{m}: 404")
                continue
            self.calls.append(messages)
            content, tcs = self.script.pop(0)
            return LLMResponse(content, tcs, 100, 20, m, {"fallback_errors": errs})
        raise ModelUnavailable("; ".join(errs))


ORDER_SCRIPT = [("", [call("lookup", {"query": "order 1042"})]),
                ("", [call("calculator", {"expression": "3*250"})]),
                ("FINAL: 750", [])]


class Tools(unittest.TestCase):
    def test_calculator_is_safe(self):
        self.assertEqual(tool_calculator("3*250+12"), "762")
        with self.assertRaises(Exception):
            tool_calculator("__import__('os').system('echo hi')")

    def test_lookup_and_convert(self):
        self.assertIn("Lena Ortiz", tool_lookup("Globex"))
        self.assertEqual(tool_convert(104, "F", "C"), "40.0")
        self.assertEqual(tool_convert(42, "km", "mi"), "26.098")

    def test_checkers(self):
        self.assertTrue(BY_ID["order_total"].check("The total is 750 INR"))
        self.assertFalse(BY_ID["order_total"].check("BANANA"))
        self.assertTrue(BY_ID["ceo"].check("Lena Ortiz, Austin"))
        self.assertEqual(len(TASKS), 8)


class Metrics(unittest.TestCase):
    def test_normal_run_counts_everything(self):
        r = run_one(AGENTS["tool_calling"], BY_ID["order_total"],
                    FakeClient(ORDER_SCRIPT), ["m1"], "normal")
        self.assertTrue(r["success"])
        self.assertEqual((r["steps"], r["tool_calls"], r["tokens"]), (3, 2, 360))
        self.assertEqual([e["kind"] for e in r["trace"]],
                         ["llm", "tool", "llm", "tool", "llm"])

    def test_failing_tool_is_injected_once_per_tool(self):
        script = [("", [call("lookup", {"query": "order 1042"})]),
                  ("", [call("lookup", {"query": "order 1042"})]), *ORDER_SCRIPT[1:]]
        r = run_one(AGENTS["tool_calling"], BY_ID["order_total"],
                    FakeClient(script), ["m1"], "failing_tool")
        self.assertEqual(r["tool_errors"], 2)  # first lookup and first calculator
        faults = [e for e in r["trace"] if e.get("injected_fault")]
        self.assertEqual({e["name"] for e in faults}, {"lookup", "calculator"})

    def test_model_outage_falls_back_and_is_recorded(self):
        r = run_one(AGENTS["tool_calling"], BY_ID["order_total"],
                    FakeClient(ORDER_SCRIPT), ["m1"], "model_outage")
        self.assertTrue(r["success"])
        self.assertEqual(r["fallbacks"], 3)

    def test_total_outage_is_a_failure_not_a_crash(self):
        c = FakeClient([], bad=("unavailable/model-does-not-exist", "m1"))
        r = run_one(AGENTS["no_tools"], BY_ID["ceo"], c, ["m1"], "model_outage")
        self.assertFalse(r["success"])
        self.assertTrue(r["error"].startswith("model_unavailable"))

    def test_misleading_injection_reaches_the_agent_and_is_detected(self):
        script = [("", [call("lookup", {"query": "globex"})]), ("FINAL: BANANA", [])]
        c = FakeClient(script)
        r = run_one(AGENTS["tool_calling"], BY_ID["ceo"], c, ["m1"], "misleading")
        self.assertIn("SYSTEM NOTICE", c.calls[1][-1]["content"])
        self.assertTrue(r["followed_injection"])
        self.assertFalse(r["success"])

    def test_step_limit_stops_looping_agents(self):
        script = [("", [call("lookup", {"query": "x"}, i)]) for i in range(20)]
        r = run_one(AGENTS["tool_calling"], BY_ID["ceo"], FakeClient(script), ["m1"],
                    "normal", max_steps=4)
        self.assertEqual(r["steps"], 4)
        self.assertTrue(r["error"].startswith("step_limit"))

    def test_text_react_and_plan_execute(self):
        s1 = [('ACTION: lookup {"query": "order 1042"}', []),
              ('ACTION: calculator {"expression": "3*250"}', []), ("FINAL: 750", [])]
        self.assertTrue(run_one(AGENTS["text_react"], BY_ID["order_total"],
                                FakeClient(s1), ["m1"], "normal")["success"])
        s2 = [('[{"tool":"lookup","args":{"query":"order 1042"}},'
               '{"tool":"calculator","args":{"expression":"3*250"}}]', []), ("FINAL: 750", [])]
        r = run_one(AGENTS["plan_execute"], BY_ID["order_total"], FakeClient(s2), ["m1"],
                    "failing_tool")
        self.assertTrue(r["success"])
        self.assertEqual(r["tool_calls"], 4)  # each tool retried once after the fault

    def test_summary(self):
        rows = [run_one(AGENTS["tool_calling"], BY_ID["order_total"],
                        FakeClient(list(ORDER_SCRIPT)), ["m1"], "normal") for _ in range(3)]
        s = summarise(rows)[0]
        self.assertEqual((s["runs"], s["success_rate"], s["median_steps"]), (3, 1.0, 3))


if __name__ == "__main__":
    unittest.main()
