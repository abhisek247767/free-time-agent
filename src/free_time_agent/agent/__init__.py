"""Router model picks tools (Phase 2), Qwen writes the answer (Phase 3)."""

from dataclasses import dataclass

from .decision_log import accuracy, log_decision, mark_correct
from .responder import respond
from .router import LoopResult, run_tool_loop


@dataclass
class Plan:
    answer: str
    templated: bool
    loop: LoopResult
    decision_id: int


def plan(request: str) -> Plan:
    loop = run_tool_loop(request)
    decision_id = log_decision(loop)
    reply = respond(loop)
    return Plan(answer=reply.text, templated=reply.templated, loop=loop, decision_id=decision_id)


__all__ = ["Plan", "accuracy", "log_decision", "mark_correct", "plan", "respond", "run_tool_loop"]
