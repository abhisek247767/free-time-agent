"""Steps 12-13: local Qwen turns the tool results into a 10-second answer."""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from free_time_agent.tools._common import env_float, env_str

from .router import OLLAMA_BASE_URL, LoopResult

RESPONSE_MODEL = env_str("RESPONSE_MODEL", "qwen2.5:3b")
RESPONSE_TEMPERATURE = env_float("RESPONSE_TEMPERATURE", 0.3)

COMMON_RULES = """You help someone plan a short walk. They will read your reply in 10 seconds.
Plain text only: no labels, lists, headings or emojis.
Copy times, places, distances and numbers from FACTS and TOOL RESULTS exactly; never calculate or invent anything.
If a tool result has an error, say in one sentence what you could not find."""

# Only the instructions for the chosen action are sent: given every action's
# rules at once, the 3B model copied examples from the wrong one.
ACTION_RULES = {
    "plan_walk": """Reply with at most 3 sentences:
1. Go to <FACTS.park.name> (<FACTS.park.distance_km> km), leave by <FACTS.leave_by> and turn back by <FACTS.turn_back_by>.
2. Carry <every item in FACTS.carry>.
3. Only if FACTS.safety is not empty: "Heads up: <the FACTS.safety items>."
If FACTS.safety is empty, stop after sentence 2.
If FACTS has no park, sentence 1 is instead: "No park found nearby, so walk around your area, leaving by <FACTS.leave_by> and turning back by <FACTS.turn_back_by>.\"""",
    "weather": "Answer the user's weather question in 1-2 sentences using the weather numbers.",
    "daylight": "Answer the user's sunrise/sunset/golden hour question in 1-2 sentences using the sun times.",
    "other": "Say in one sentence that you can only help plan walks and check weather and daylight.",
}

MAX_SENTENCES = 3
# Split after . ! ? followed by a space and a capital letter; "17:30" and "2.5 km" are safe.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?)\b", re.IGNORECASE)
# Small models sometimes narrate the rules ("No third note needed..."); drop those sentences.
_META_SENTENCE = re.compile(
    r"\b(third (sentence|note)|no (safety )?(note|warning)s?\b|(note|warning) (is )?not needed)", re.IGNORECASE
)


def make_responder() -> ChatOllama:
    return ChatOllama(
        model=RESPONSE_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=RESPONSE_TEMPERATURE,
    )


def _parse_free_minutes(request: str) -> int | None:
    match = _DURATION_RE.search(request)
    if not match:
        return None
    amount, unit = float(match.group(1)), match.group(2).lower()
    return round(amount * 60) if unit.startswith("h") else round(amount)


def _hhmm(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def build_facts(result: LoopResult, now: datetime | None = None) -> dict:
    """Times, weather summary, what to carry and safety flags, computed in code.

    A 3B model is unreliable at time arithmetic and threshold checks, so the
    responder only has to phrase these facts, not derive them.
    """
    now = now or datetime.now()
    leave_by = now + timedelta(minutes=-now.minute % 5 or 5)
    facts: dict = {"now": _hhmm(now), "leave_by": _hhmm(leave_by)}
    safety: list[str] = []
    carry = ["water"]

    weather = result.output_of("get_weather")
    if weather and weather["hours"]:
        hours = weather["hours"]
        rain = max(h["rain_chance_pct"] or 0 for h in hours)
        temp = max(h["temperature_c"] for h in hours)
        wind = max(h["wind_kmh"] for h in hours)
        facts["weather"] = {"max_rain_chance_pct": rain, "max_temp_c": temp, "max_wind_kmh": wind}
        if temp >= 30:
            carry.append("a cap")
        if rain >= 30:
            carry.append("an umbrella")
        if rain >= 40:
            safety.append(f"{rain}% chance of rain")
        if wind > 30:
            safety.append(f"strong wind up to {wind} km/h")

    parks = (result.output_of("find_green_spaces") or {}).get("places")
    route = result.output_of("get_walk_route")
    if parks:
        park = {"name": parks[0]["name"]}
        if route:
            park["distance_km"] = route["distance_km"]
        else:
            park["distance_km"] = round(parks[0]["distance_m"] / 1000, 2)
            if route.get("duration_min"):
                park["walk_min_one_way"] = route["duration_min"]
        facts["park"] = park

    sun = result.output_of("get_sun_times")
    if sun:
        facts["sunset"] = sun["sunset"]
        sunset = datetime.combine(now.date(), datetime.strptime(sun["sunset"], "%H:%M").time())
        free_min = _parse_free_minutes(result.request)
        one_way = (facts.get("park") or {}).get("walk_min_one_way", 15)
        walk_min = free_min or 2 * one_way + 30
        home = leave_by + timedelta(minutes=walk_min)

        turnaround = result.output_of("turnaround_time")
        latest_turn = (
            datetime.combine(now.date(), datetime.strptime(turnaround["turnaround_at"], "%H:%M").time())
            if turnaround and turnaround.get("turnaround_at")
            else None
        )
        turn_back = leave_by + timedelta(minutes=walk_min / 2)
        no_daylight = (turnaround and not turnaround.get("safe")) or (
            latest_turn and latest_turn <= leave_by + timedelta(minutes=5)
        )
        if no_daylight:
            safety.append(f"not enough daylight left: sunset is at {sun['sunset']}")
            facts["carry"] = carry
            facts["safety"] = safety
            return facts
        if latest_turn and latest_turn < turn_back:
            turn_back = latest_turn
            home = leave_by + 2 * (turn_back - leave_by)
        facts["turn_back_by"] = _hhmm(turn_back)
        facts["home_by"] = _hhmm(home)

        if home > sunset - timedelta(minutes=30):
            safety.append(f"you will be heading home close to sunset at {sun['sunset']}")
            carry.append("a phone torch")

    facts["carry"] = carry
    facts["safety"] = safety
    return facts


def build_context(result: LoopResult, facts: dict) -> str:
    context = f"User request: {result.request}\nAction: {result.intent}\n\nFACTS:\n{json.dumps(facts, indent=1)}"
    # Walk plans get FACTS only: given the raw turnaround output too, the model
    # copied its "back_by" as the turn-back time.
    if result.intent != "plan_walk":
        tool_results = [{"tool": r.name, "result": r.output} for r in result.runs]
        context += f"\n\nTOOL RESULTS:\n{json.dumps(tool_results, indent=1, default=str)}"
    return context


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def template_walk_answer(facts: dict) -> str:
    """Deterministic answer built straight from FACTS, used when the model's reply fails checks."""
    if "turn_back_by" not in facts:
        return f"Skip the walk for now: {_join(facts['safety'])}." if facts["safety"] else "I couldn't plan a walk."
    when = f"leave by {facts['leave_by']} and turn back by {facts['turn_back_by']}"
    park = facts.get("park")
    if park:
        first = f"Go to {park['name']} ({park['distance_km']} km), {when}."
    else:
        first = f"No park found nearby, so walk around your area: {when}."
    parts = [first, f"Carry {_join(facts['carry'])}."]
    if facts["safety"]:
        parts.append(f"Heads up: {_join(facts['safety'])}.")
    return " ".join(parts)


def _walk_answer_ok(text: str, facts: dict) -> bool:
    """The reply must contain the key facts verbatim and no unfilled placeholders."""
    if "<" in text or ">" in text or "turn_back_by" not in facts:
        return False
    required = [facts["leave_by"], facts["turn_back_by"]]
    park = facts.get("park")
    if park:
        required.append(park["name"])
    elif "no park" not in text.lower():
        return False
    return all(item in text for item in required)


@dataclass
class Reply:
    text: str
    templated: bool  # True when the model's reply failed checks and the template was used


def respond(result: LoopResult, responder=None) -> Reply:
    responder = responder or make_responder()
    facts = build_facts(result)
    system = f"{COMMON_RULES}\n\n{ACTION_RULES.get(result.intent, ACTION_RULES['other'])}"
    reply = responder.invoke([SystemMessage(system), HumanMessage(build_context(result, facts))])
    sentences = [s for s in _SENTENCE_END.split(reply.content.strip()) if not _META_SENTENCE.search(s)]
    # The safety sentence is only allowed when code found a real risk.
    limit = MAX_SENTENCES if facts["safety"] or result.intent != "plan_walk" else MAX_SENTENCES - 1
    text = " ".join(sentences[:limit])

    if result.intent == "plan_walk" and not _walk_answer_ok(text, facts):
        return Reply(template_walk_answer(facts), templated=True)
    return Reply(text, templated=False)
