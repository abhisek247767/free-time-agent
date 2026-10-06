"""Step 11: log every router decision so tool-choice accuracy can be measured.

`correct` is NULL until it is judged, either by the user in the CLI or
automatically against a labelled expected intent (scripts/eval_router.py).
"""

import json
import sqlite3
from datetime import datetime

from free_time_agent.tools._common import cache_path

from .router import ROUTER_MODEL, LoopResult


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(cache_path("router_log.db"))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at TEXT NOT NULL,
            model TEXT NOT NULL,
            request TEXT NOT NULL,
            intent TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities TEXT NOT NULL,
            tool_calls TEXT NOT NULL,
            expected TEXT,
            correct INTEGER
        )"""
    )
    return conn


def log_decision(result: LoopResult, expected: str | None = None) -> int:
    calls = [{"round": r.round, "name": r.name, "args": r.args, "ok": "error" not in r.output} for r in result.runs]
    correct = None if expected is None else int(result.intent == expected)
    with _db() as conn:
        cur = conn.execute(
            """INSERT INTO decisions
               (logged_at, model, request, intent, confidence, probabilities, tool_calls, expected, correct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(timespec="seconds"),
                ROUTER_MODEL,
                result.request,
                result.intent,
                result.confidence,
                json.dumps(result.probabilities),
                json.dumps(calls),
                expected,
                correct,
            ),
        )
    return cur.lastrowid


def mark_correct(decision_id: int, correct: bool) -> None:
    with _db() as conn:
        conn.execute("UPDATE decisions SET correct = ? WHERE id = ?", (int(correct), decision_id))


def accuracy(model: str | None = None) -> dict:
    """Accuracy over all judged decisions, optionally for one router model."""
    where, params = ("WHERE model = ?", (model,)) if model else ("", ())
    with _db() as conn:
        total, judged, correct, avg_conf = conn.execute(
            f"""SELECT COUNT(*), COUNT(correct), COALESCE(SUM(correct), 0), AVG(confidence)
                FROM decisions {where}""",
            params,
        ).fetchone()
    return {
        "decisions": total,
        "judged": judged,
        "correct": correct,
        "accuracy_pct": round(100 * correct / judged, 1) if judged else None,
        "avg_confidence": round(avg_conf, 3) if avg_conf is not None else None,
    }
