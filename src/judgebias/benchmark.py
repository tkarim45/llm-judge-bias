"""Run the judge on every pair in BOTH orders and turn the choices into bias numbers.

For each pair we hold two texts: `better` and `worse`. What "worse" means depends on the bias:
- position: `worse` is genuinely incorrect — a fair judge never picks it (pick-worse rate ≈ 0).
- verbosity: `worse` is the longer (equally-correct) answer — a fair judge is indifferent (≈ 0.5).
- assertiveness: `worse` is the confident-phrased (equally-correct) answer — fair ≈ 0.5.

Presenting each pair in both slot orders lets us separate two things: how often the judge prefers the
*treatment* (wrong / longer / confident), and how often it just prefers whichever answer is in the
first slot (position bias). A fair judge's verdict is invariant to order; a biased one flips.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .data import Pair, all_pairs

JUDGE_SYSTEM = (
    "You are comparing two candidate answers, A and B, to a question. Decide which answer is better. "
    "Reply with exactly one character — 'A' or 'B' — and nothing else."
)


def _prompt(question: str, a: str, b: str) -> str:
    return f"Question: {question}\n\nAnswer A: {a}\n\nAnswer B: {b}\n\nWhich answer is better, A or B?"


def parse_choice(text: str) -> str | None:
    m = re.search(r"\b([AB])\b", text.strip())
    return m.group(1).upper() if m else None


@dataclass
class BiasFakeJudge:
    """Offline judge for tests. Deterministic and configurably biased:
    - always_first: pick whatever is in slot A (pure position bias)
    - prefer_longer: pick the longer text (verbosity bias)
    Default picks the longer text, which exercises both the verbosity metric and the order logic."""
    mode: str = "prefer_longer"

    def complete(self, system: str, prompt: str):
        from .llm import Reply
        a = prompt.split("Answer A:", 1)[1].split("Answer B:", 1)[0]
        b = prompt.split("Answer B:", 1)[1].split("Which answer", 1)[0]
        if self.mode == "always_first":
            choice = "A"
        else:
            choice = "A" if len(a) >= len(b) else "B"
        return Reply(choice, len(prompt.split()), 1)


def _judge_once(client, question: str, slot_a: str, slot_b: str):
    reply = client.complete(JUDGE_SYSTEM, _prompt(question, slot_a, slot_b))
    return parse_choice(reply.text), reply


def run(client, pairs: list[Pair] | None = None) -> dict:
    items = pairs if pairs is not None else all_pairs()
    by_bias: dict[str, dict] = {}
    in_tok = out_tok = 0
    records = []

    for p in items:
        stats = by_bias.setdefault(p.bias, {"judgments": 0, "picked_worse": 0,
                                            "picked_first_slot": 0, "flips": 0, "pairs": 0})
        # order 1: better in A, worse in B ; order 2: worse in A, better in B
        c1, r1 = _judge_once(client, p.question, p.better, p.worse)
        c2, r2 = _judge_once(client, p.question, p.worse, p.better)
        in_tok += r1.input_tokens + r2.input_tokens
        out_tok += r1.output_tokens + r2.output_tokens

        # map slot choice -> which logical text was picked ("better"/"worse")
        pick1 = "better" if c1 == "A" else "worse" if c1 == "B" else None
        pick2 = "worse" if c2 == "A" else "better" if c2 == "B" else None

        for pick, choice in ((pick1, c1), (pick2, c2)):
            if pick is None:
                continue
            stats["judgments"] += 1
            if pick == "worse":
                stats["picked_worse"] += 1
            if choice == "A":
                stats["picked_first_slot"] += 1
        stats["pairs"] += 1
        if pick1 is not None and pick2 is not None and pick1 != pick2:
            stats["flips"] += 1
        records.append({"id": p.id, "bias": p.bias, "order1_pick": pick1, "order2_pick": pick2,
                        "flipped": (pick1 != pick2) if (pick1 and pick2) else None})

    summary = {}
    for bias, s in by_bias.items():
        j = s["judgments"] or 1
        summary[bias] = {
            "pairs": s["pairs"],
            "judgments": s["judgments"],
            "pick_worse_rate": round(s["picked_worse"] / j, 3),
            "first_slot_rate": round(s["picked_first_slot"] / j, 3),
            "flip_rate": round(s["flips"] / (s["pairs"] or 1), 3),
        }
    return {"by_bias": summary, "tokens": {"input": in_tok, "output": out_tok}, "records": records}


def format_report(r: dict) -> str:
    labels = {
        "position": "pick_worse = chose the WRONG answer (fair≈0.00)",
        "verbosity": "pick_worse = chose the LONGER answer (fair≈0.50)",
        "assertiveness": "pick_worse = chose the CONFIDENT phrasing (fair≈0.50)",
    }
    lines = [f"{'bias':<14} {'pairs':>5} {'pick_worse':>11} {'first_slot':>11} {'flip_rate':>10}",
             "-" * 55]
    for bias in ("position", "verbosity", "assertiveness"):
        if bias in r["by_bias"]:
            s = r["by_bias"][bias]
            lines.append(f"{bias:<14} {s['pairs']:>5} {s['pick_worse_rate']:>11.3f} "
                         f"{s['first_slot_rate']:>11.3f} {s['flip_rate']:>10.3f}")
    lines.append("")
    lines.append("first_slot_rate = how often the judge picked whichever answer was shown first "
                 "(0.50 = no position preference)")
    for bias in ("position", "verbosity", "assertiveness"):
        if bias in r["by_bias"]:
            lines.append(f"  {bias}: {labels[bias]}")
    return "\n".join(lines)
