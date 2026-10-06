"""Steps 9-10: tev1 decides which tools to run; code fills arguments and runs them.

tev1 is a decision (classification) model, not a chat/tool-calling model. It is
called through Ollama's /v1/systemone endpoint with a `state` (the user's
message) and `questions`, and returns a choice plus probabilities. It cannot
produce arguments such as a place name, so:

1. tev1 picks one action (plan_walk, weather, daylight, log_walk, other).
2. Each action maps to a fixed set of tools.
3. Arguments come from the request (regex, with Qwen as a fallback for the
   place name) and from earlier tool results (coordinates, sunset, nearest park).
4. Tools run in dependency rounds: geocode -> weather/sun/parks -> route/turnaround,
   so at most 3 rounds.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from langchain_ollama import ChatOllama

from free_time_agent.tools._common import env_float, env_str

from .lc_tools import TOOLS_BY_NAME

OLLAMA_BASE_URL = env_str("OLLAMA_BASE_URL", "http://localhost:11434")
ROUTER_MODEL = env_str("ROUTER_MODEL", "tev1:0.8b")
ROUTER_TIMEOUT_S = env_float("ROUTER_TIMEOUT_S", 120.0)
# Appended to place names for geocoding, e.g. "Hyderabad" -> "Gachibowli, Hyderabad".
HOME_CITY = env_str("HOME_CITY", "")
# Qwen is only used here to pull a place name out of the request when regex fails.
EXTRACT_MODEL = env_str("RESPONSE_MODEL", "qwen2.5:3b")

# Option wording matters a lot for a 0.8B classifier. These short "action"
# phrasings scored 10/10 on a labelled sample; longer descriptions did worse.
INTENT_QUESTION = {
    "type": "choice",
    "instructions": "Which action should a walk-planning assistant take for this user message?",
    "criteria": {
        "plan_walk": "Recommend a park or walk for the user's free time",
        "weather": "Answer a weather question",
        "daylight": "Answer a sunrise, sunset or golden hour question",
        "log_walk": "Save a walk the user finished",
        "other": "Decline: unrelated to walking",
    },
}

# Tools for each action, grouped into rounds. A later round may use earlier results.
PLAYBOOK: dict[str, list[list[str]]] = {
    "plan_walk": [
        ["geocode"],
        ["get_weather", "get_sun_times", "find_green_spaces"],
        ["get_walk_route", "turnaround_time"],
    ],
    "weather": [["geocode"], ["get_weather"]],
    "daylight": [["geocode"], ["get_sun_times"]],
    "log_walk": [["log_outing"]],
    "other": [],
}

# Capitalised words after a preposition: "near Gachibowli", "at KBR Park today".
_PLACE_RE = re.compile(r"\b(?:near|around|at|in|of|from|to)\s+([A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*)*)")
_DISTANCE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:km|kms|kilometers?|kilometres?)\b", re.IGNORECASE)


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
    probabilities: dict = field(default_factory=dict)
    runs: list[ToolRun] = field(default_factory=list)
    rounds: int = 0

    @property
    def tools_called(self) -> list[str]:
        return [r.name for r in self.runs]

    def output_of(self, name: str) -> dict | None:
        for run in self.runs:
            if run.name == name and "error" not in run.output:
                return run.output
        return None


def decide(request: str) -> dict:
    """Ask tev1 which action to take. Returns {choice, probabilities, confidence}."""
    resp = httpx.post(
        f"{OLLAMA_BASE_URL}/v1/systemone",
        json={"model": ROUTER_MODEL, "state": request, "questions": {"intent": INTENT_QUESTION}},
        timeout=ROUTER_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()["answers"]["intent"]


def extract_place(request: str) -> str | None:
    match = _PLACE_RE.search(request)
    if match:
        place = match.group(1)
    else:
        llm = ChatOllama(model=EXTRACT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
        reply = llm.invoke(
            "Extract the place or area name from this message. Reply with only the name, "
            f"or NONE if there is no place.\nMessage: {request}"
        ).content.strip().strip("\"'.")
        if not reply or reply.upper() == "NONE":
            return None
        place = reply
    if HOME_CITY and HOME_CITY.lower() not in place.lower():
        place = f"{place}, {HOME_CITY}"
    return place


def _build_args(name: str, request: str, result: LoopResult) -> dict | str:
    """Arguments for a tool, or an error string if a dependency is missing."""
    if name == "geocode":
        place = extract_place(request)
        return {"place_name": place} if place else "no place name found in the request"

    if name == "log_outing":
        distance = _DISTANCE_RE.search(request)
        if not distance:
            return "no distance in km found in the request"
        return {
            "place": extract_place(request) or "unknown",
            "distance_km": float(distance.group(1)),
            "notes": request,
        }

    here = result.output_of("geocode")
    if here is None:
        return "location unknown because geocode failed"

    if name in ("get_weather", "get_sun_times", "find_green_spaces"):
        return {"lat": here["lat"], "lon": here["lon"]}

    if name == "get_walk_route":
        parks = (result.output_of("find_green_spaces") or {}).get("places")
        if not parks:
            return "no green spaces found to route to"
        return {"from_lat": here["lat"], "from_lon": here["lon"], "to_lat": parks[0]["lat"], "to_lon": parks[0]["lon"]}

    if name == "turnaround_time":
        sun = result.output_of("get_sun_times")
        if sun is None:
            return "sunset unknown"
        return {"start_time": datetime.now().strftime("%H:%M"), "sunset": sun["sunset"]}

    return f"no argument builder for {name}"


def _execute(name: str, args: dict) -> dict:
    try:
        return TOOLS_BY_NAME[name].invoke(args)
    except Exception as exc:  # report failures to the responder instead of crashing
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_tool_loop(request: str) -> LoopResult:
    decision = decide(request)
    result = LoopResult(
        request=request,
        intent=decision["choice"],
        confidence=decision["probabilities"][decision["choice"]],
        probabilities=decision["probabilities"],
    )

    for round_no, tool_names in enumerate(PLAYBOOK.get(result.intent, []), start=1):
        result.rounds = round_no
        for name in tool_names:
            args = _build_args(name, request, result)
            output = {"error": args} if isinstance(args, str) else _execute(name, args)
            result.runs.append(ToolRun(round_no, name, {} if isinstance(args, str) else args, output))

    return result
