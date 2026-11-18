"""Benchmark tasks. Each task has a deterministic checker on the final answer."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


def _num(ans: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", ans)]


def has_number(target: float, tol: float = 0.01) -> Callable[[str], bool]:
    return lambda a: any(abs(n - target) <= tol * max(1, abs(target)) for n in _num(a))


def has_text(*words: str) -> Callable[[str], bool]:
    return lambda a: all(w.lower() in a.lower() for w in words)


@dataclass
class Task:
    id: str
    prompt: str
    check: Callable[[str], bool]
    needs: tuple  # tools a good agent should use


TASKS = [
    Task("order_total", "What is the total value in INR of order 1042? Reply with the number.",
         has_number(750), ("lookup", "calculator")),
    Task("two_orders", "What is the combined value in INR of orders 1042 and 2087?",
         has_number(1350), ("lookup", "calculator")),
    Task("ceo", "Who is the CEO of Globex, and in which city is it headquartered?",
         has_text("Lena Ortiz", "Austin"), ("lookup",)),
    Task("headcount_diff", "How many more employees does Globex have than Acme Corp?",
         has_number(1480), ("lookup", "calculator")),
    Task("distance", "A delivery route is 42 km. How many miles is that? Round to 1 decimal.",
         has_number(26.1, 0.005), ("convert",)),
    Task("temp", "The warehouse is at 104 F. What is that in Celsius?",
         has_number(40), ("convert",)),
    Task("refund", "A customer received an unused item 20 days ago. Can they get a refund? "
         "Answer yes or no and give the policy window.", has_text("yes", "30"), ("lookup",)),
    Task("compound", "Order 2087's value grows 10% per year. What is it after 2 years, in INR?",
         has_number(726), ("lookup", "calculator")),
]

BY_ID = {t.id: t for t in TASKS}
