"""Steps 9-10: ROUTER_MODEL decides which tools; RESPONSE_MODEL fills their arguments.

1. tev1 (a decision model, called via Ollama's /v1/systemone) picks one action.
   Each action maps to a fixed list of tools. That is its only job.
2. Qwen gets only those tools via bind_tools and writes the tool calls with
   arguments, round by round, seeing each round's results. Stops when it makes
   no more calls or after MAX_ROUNDS.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from free_time_agent.tools._common import env_float, env_int, env_str

from .lc_tools import TOOLS_BY_NAME

OLLAMA_BASE_URL = env_str("OLLAMA_BASE_URL", "http://localhost:11434")
ROUTER_MODEL = env_str("ROUTER_MODEL", "tev1:0.8b")
ROUTER_TIMEOUT_S = env_float("ROUTER_TIMEOUT_S", 120.0)
RESPONSE_MODEL = env_str("RESPONSE_MODEL", "qwen2.5:3b")
MAX_ROUNDS = env_int("MAX_TOOL_ROUNDS", 3)
HOME_CITY = env_str("HOME_CITY", "")

# Option wording matters a lot for a 0.8B classifier; these short phrasings
# scored best on a labelled sample.
DECISION_QUESTION = {
    "type": "choice",
    "instructions": "Which action should a walk-planning assistant take for this user message?",
    "criteria": {
        "plan_walk": "Recommend a park or walk for the user's free time",
        "weather": "Answer a weather question",
        "daylight": "Answer a sunrise, sunset or golden hour question",
        "other": "Decline: unrelated to walking",
    },
}

TOOLS_FOR_ACTION = {
    "plan_walk": ["geocode", "get_weather", "get_sun_times", "find_green_spaces", "get_walk_route", "turnaround_time"],
    "weather": ["geocode", "get_weather"],
    "daylight": ["geocode", "get_sun_times"],
    "other": [],
}

ARGS_PROMPT = """You fill in arguments for tool calls. Do not answer the user.
Now: {now}.{city_hint}
Tools to use: {tools}.
- First call geocode ALONE with the place name from the user's message.
- Then call the other tools using the exact lat/lon geocode returned.
- turnaround_time: start_time is the current time, sunset comes from get_sun_times.
- get_walk_route: from the geocode lat/lon to the first park from find_green_spaces.
When every tool has been called, reply DONE with no tool calls."""


@dataclass
class ToolRun:
    round: int
    name: str
    args: dict
    output: dict


@dataclass
class LoopResult:
    request: str
    intent: str = "other"
    confidence: float = 0.0
    runs: list[ToolRun] = field(default_factory=list)
    rounds: int = 0

    @property
    def tools_called(self) -> list[str]:
        return [r.name for r in self.runs]

    def output_of(self, name: str) -> dict | None:
        for run in reversed(self.runs):
            if run.name == name and "error" not in run.output:
                return run.output
        return None


def decide(request: str) -> dict:
    """Ask the router model which action to take. Returns {choice, probabilities, confidence}."""
    resp = httpx.post(
        f"{OLLAMA_BASE_URL}/v1/systemone",
        json={"model": ROUTER_MODEL, "state": request, "questions": {"action": DECISION_QUESTION}},
        timeout=ROUTER_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()["answers"]["action"]


def _execute(name: str, args: dict) -> dict:
    try:
        return TOOLS_BY_NAME[name].invoke(args)
    except Exception as exc:  # report failures back to the model instead of crashing
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_tool_loop(request: str) -> LoopResult:
    decision = decide(request)
    result = LoopResult(
        request=request,
        intent=decision["choice"],
        confidence=decision["probabilities"][decision["choice"]],
    )
    tool_names = TOOLS_FOR_ACTION.get(result.intent, [])
    if not tool_names:
        return result

    llm = ChatOllama(model=RESPONSE_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    llm = llm.bind_tools([TOOLS_BY_NAME[n] for n in tool_names])
    city_hint = f"\nIf the user gives only an area name, add \", {HOME_CITY}\" to it." if HOME_CITY else ""
    system = ARGS_PROMPT.format(
        now=datetime.now().strftime("%A %Y-%m-%d %H:%M"), city_hint=city_hint, tools=", ".join(tool_names)
    )
    messages = [SystemMessage(system), HumanMessage(request)]

    for round_no in range(1, MAX_ROUNDS + 1):
        ai = llm.invoke(messages)
        if not ai.tool_calls:
            break
        result.rounds = round_no
        messages.append(ai)
        # Coordinates guessed in the same round as geocode are made up, so
        # geocode runs alone and the rest are asked for again next round.
        has_geocode = any(c["name"] == "geocode" for c in ai.tool_calls)
        for call in ai.tool_calls:
            if call["name"] not in tool_names:
                output = {"error": f"{call['name']} is not allowed for this request"}
            elif has_geocode and call["name"] != "geocode":
                output = {"error": "skipped: call again next round with geocode's lat/lon"}
            else:
                output = _execute(call["name"], call["args"])
                result.runs.append(ToolRun(round_no, call["name"], call["args"], output))
            messages.append(ToolMessage(json.dumps(output, default=str), tool_call_id=call["id"], name=call["name"]))

    return result
