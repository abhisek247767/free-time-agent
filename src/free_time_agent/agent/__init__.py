"""Router model picks tools (Phase 2), Qwen fills arguments and writes the answer (Phase 3)."""

from collections.abc import Callable
from dataclasses import dataclass

from .responder import respond
from .router import DEFAULT_LOCATION, LoopResult, run_tool_loop


@dataclass
class Plan:
    answer: str
    headline: str
    lines: list[str]
    templated: bool
    loop: LoopResult


def plan(
    request: str, location: str = DEFAULT_LOCATION, on_step: Callable[[str], None] | None = None
) -> Plan:
    loop = run_tool_loop(request, location, on_step)
    reply = respond(loop)
    return Plan(
        answer=reply.text, headline=reply.headline, lines=reply.lines, templated=reply.templated, loop=loop
    )


__all__ = ["DEFAULT_LOCATION", "Plan", "plan", "respond", "run_tool_loop"]
