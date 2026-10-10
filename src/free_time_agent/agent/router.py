"""Steps 9-10: ROUTER_MODEL decides which tools; RESPONSE_MODEL fills their arguments.

1. tev1 (a decision model, called via Ollama's /v1/systemone) picks one action.
   Each action maps to a fixed list of tools. That is its only job.
2. Qwen gets only those tools via bind_tools and writes the tool calls with
   arguments, round by round, seeing each round's results. Stops when it makes
   no more calls or after MAX_ROUNDS.
"""

import json
from collections.abc import Callable
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
DEFAULT_LOCATION = env_str("DEFAULT_LOCATION", "Kolkata, West Bengal")

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
Now: {now}.
Tools to use: {tools}.
- First call geocode ALONE with the place named in the user's message.
  Only if the message names no place at all, use "{location}".
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
    location: str = DEFAULT_LOCATION
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


def _region(location: str) -> str:
    """Last part of the location, e.g. "West Bengal", used to disambiguate place names."""
    return location.split(",")[-1].strip()


def _geocode(args: dict, location: str) -> dict:
    """Look the place up within the user's region first, then anywhere."""
    place, region = args["place_name"], _region(location)
    if region and region.lower() not in place.lower():
        hit = _execute("geocode", {"place_name": f"{place}, {region}"})
        if "error" not in hit:
            return hit
    return _execute("geocode", args)


def _checked_args(name: str, args: dict, location: str, result: LoopResult) -> dict:
    """Keep Qwen's arguments, but replace values a 3B model is known to get wrong.

    Coordinates must come from tool results, never from the model's guesses.
    """
    if name == "geocode":
        if not args.get("place_name"):
            args = {"place_name": location}  # it sometimes passes lat/lon instead of a name
        return {"place_name": args["place_name"]}

    here = result.output_of("geocode")
    if here:
        for key, src in (("lat", "lat"), ("lon", "lon"), ("from_lat", "lat"), ("from_lon", "lon")):
            if key in args or (name == "get_walk_route" and key.startswith("from")):
                args[key] = here[src]
    if name == "get_walk_route":
        parks = (result.output_of("find_green_spaces") or {}).get("places") or []
        if parks and (args.get("to_lat"), args.get("to_lon")) not in {(p["lat"], p["lon"]) for p in parks}:
            args["to_lat"], args["to_lon"] = parks[0]["lat"], parks[0]["lon"]
    if name in ("get_weather", "get_sun_times", "find_green_spaces") and here:
        args.setdefault("lat", here["lat"])
        args.setdefault("lon", here["lon"])
    return args


def run_tool_loop(
    request: str, location: str = DEFAULT_LOCATION, on_step: Callable[[str], None] | None = None
) -> LoopResult:
    """Run router + tool rounds. `on_step` receives short progress messages (used by the UI)."""
    step = on_step or (lambda _msg: None)
    decision = decide(request)
    result = LoopResult(
        request=request,
        location=location,
        intent=decision["choice"],
        confidence=decision["probabilities"][decision["choice"]],
    )
    tool_names = TOOLS_FOR_ACTION.get(result.intent, [])
    step(f"Router chose **{result.intent}** ({result.confidence:.0%}): {', '.join(tool_names) or 'no tools'}")
    if not tool_names:
        return result

    llm = ChatOllama(model=RESPONSE_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    llm = llm.bind_tools([TOOLS_BY_NAME[n] for n in tool_names])
    system = ARGS_PROMPT.format(
        now=datetime.now().strftime("%A %Y-%m-%d %H:%M"), location=location, tools=", ".join(tool_names)
    )
    messages = [SystemMessage(system), HumanMessage(request)]

    for round_no in range(1, MAX_ROUNDS + 1):
        step(f"Round {round_no}: {RESPONSE_MODEL} is choosing tool arguments…")
        ai = llm.invoke(messages)
        if not ai.tool_calls:
            missing = [n for n in tool_names if n not in result.tools_called]
            if not missing:
                break
            # The small model sometimes stops early; remind it once per round.
            messages += [ai, HumanMessage(f"You have not called: {', '.join(missing)}. Call them now.")]
            continue
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
            elif call["name"] == "geocode" and result.output_of("geocode"):
                output = result.output_of("geocode")  # already located; don't look up again
            else:
                args = _checked_args(call["name"], dict(call["args"]), location, result)
                output = _geocode(args, location) if call["name"] == "geocode" else _execute(call["name"], args)
                step(f"Ran `{call['name']}`" + (" (error)" if "error" in output else ""))
                result.runs.append(ToolRun(round_no, call["name"], args, output))
            messages.append(ToolMessage(json.dumps(output, default=str), tool_call_id=call["id"], name=call["name"]))

    return result
