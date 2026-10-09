"""Router model picks tools (Phase 2), Qwen fills arguments and writes the answer (Phase 3)."""

from dataclasses import dataclass

from .responder import respond
from .router import LoopResult, run_tool_loop


@dataclass
class Plan:
    answer: str
    templated: bool
    loop: LoopResult


def plan(request: str) -> Plan:
    loop = run_tool_loop(request)
    reply = respond(loop)
    return Plan(answer=reply.text, templated=reply.templated, loop=loop)


__all__ = ["Plan", "plan", "respond", "run_tool_loop"]
