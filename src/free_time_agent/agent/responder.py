"""Steps 12-13: local Qwen writes a one-line headline; code adds compact fact lines.

The reply is a short headline sentence from the model plus 3-5 detail lines
(park, times, weather, what to carry, warnings) built in code from the tool
results, so every number shown is exact.
"""

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

COMMON_RULES = """You write the one-line headline of a walk-planner reply.
Write exactly ONE short, friendly sentence of at most 20 words.
Plain text only: no labels, lists, emojis or quotes.
The details (times, distances, numbers) are shown below your sentence, so do not repeat them unless told to.
Never invent places or facts that are not in FACTS."""

# Only the instructions for the chosen action are sent: given every action's
# rules at once, the 3B model copied examples from the wrong one.
ACTION_RULES = {
    "plan_walk": "Sum up the plan: name FACTS.park.name and why it suits now (weather or time left). "
    "Do not write any numbers or times. If FACTS has no park, suggest a walk around the neighbourhood.",
    "weather": "Answer the user's weather question directly, using the numbers in FACTS.weather.",
    "daylight": "Answer the user's sunrise/sunset/golden hour question directly, using the times in FACTS.",
    "other": "Say that you can only help plan walks and check weather and daylight.",
}

# Split after . ! ? followed by a space and a capital letter; "17:30" and "2.5 km" are safe.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?)\b", re.IGNORECASE)


def make_responder() -> ChatOllama:
    return ChatOllama(model=RESPONSE_MODEL, base_url=OLLAMA_BASE_URL, temperature=RESPONSE_TEMPERATURE)


def _parse_free_minutes(request: str) -> int | None:
    match = _DURATION_RE.search(request)
    if not match:
        return None
    amount, unit = float(match.group(1)), match.group(2).lower()
    return round(amount * 60) if unit.startswith("h") else round(amount)


def _hhmm(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def _at(now: datetime, hhmm: str) -> datetime:
    return datetime.combine(now.date(), datetime.strptime(hhmm, "%H:%M").time())


def build_facts(result: LoopResult, now: datetime | None = None) -> dict:
    """Times, weather summary, what to carry and safety flags, computed in code.

    A 3B model is unreliable at time arithmetic and threshold checks, so these
    are derived here and the model only has to phrase the headline.
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
        facts["weather"] = {
            "hours": len(hours),
            "min_temp_c": min(h["temperature_c"] for h in hours),
            "max_temp_c": temp,
            "max_rain_chance_pct": rain,
            "max_wind_kmh": wind,
        }
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
            if route.get("duration_min"):
                park["walk_min_one_way"] = route["duration_min"]
        else:
            park["distance_km"] = round(parks[0]["distance_m"] / 1000, 2)
        facts["park"] = park

    sun = result.output_of("get_sun_times")
    if sun:
        facts.update(sunrise=sun["sunrise"], sunset=sun["sunset"], golden_hour_evening=sun["golden_hour_evening"])

    if sun and result.intent == "plan_walk":
        sunset = _at(now, sun["sunset"])
        free_min = _parse_free_minutes(result.request)
        one_way = (facts.get("park") or {}).get("walk_min_one_way", 15)
        walk_min = free_min or 2 * one_way + 30
        home = leave_by + timedelta(minutes=walk_min)
        turn_back = leave_by + timedelta(minutes=walk_min / 2)

        turnaround = result.output_of("turnaround_time")
        latest_turn = _at(now, turnaround["turnaround_at"]) if turnaround and turnaround.get("turnaround_at") else None
        no_daylight = (turnaround and not turnaround.get("safe")) or (
            latest_turn and latest_turn <= leave_by + timedelta(minutes=5)
        )
        if no_daylight:
            safety.append(f"not enough daylight left: sunset is at {sun['sunset']}")
        else:
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


def detail_lines(intent: str, facts: dict) -> list[str]:
    """Compact fact lines shown under the headline."""
    lines: list[str] = []
    w = facts.get("weather")

    if intent == "plan_walk":
        park = facts.get("park")
        if park:
            if park["distance_km"] < 0.05:
                lines.append(f"🌳 {park['name']} · right where you are")
            else:
                walk = f", about {park['walk_min_one_way']} min on foot" if park.get("walk_min_one_way") else ""
                lines.append(f"🌳 {park['name']} · {park['distance_km']} km{walk}")
        else:
            lines.append("🌳 No park found nearby · walk around your area")
        if "turn_back_by" in facts:
            lines.append(f"🕒 Leave {facts['leave_by']} → turn back {facts['turn_back_by']} → home by {facts['home_by']}")
        if w:
            sunset = f" · sunset {facts['sunset']}" if "sunset" in facts else ""
            lines.append(f"🌤 {w['max_temp_c']}°C · {w['max_rain_chance_pct']}% rain · wind {w['max_wind_kmh']} km/h{sunset}")
        lines.append(f"🎒 Carry {_join(facts['carry'])}")
    elif intent == "weather" and w:
        lines.append(
            f"🌤 Next {w['hours']} h: {w['min_temp_c']}–{w['max_temp_c']}°C · rain up to {w['max_rain_chance_pct']}%"
            f" · wind up to {w['max_wind_kmh']} km/h"
        )
        if len(facts["carry"]) > 1:
            lines.append(f"🎒 Carry {_join(facts['carry'])}")
    elif intent == "daylight" and "sunset" in facts:
        lines.append(f"🌅 Sunrise {facts['sunrise']} · Sunset {facts['sunset']}")
        lines.append(f"✨ Evening golden hour {facts['golden_hour_evening']}")

    for warning in facts["safety"]:
        lines.append(f"⚠️ {warning[0].upper()}{warning[1:]}")
    return lines


def fallback_headline(intent: str, facts: dict) -> str:
    if intent == "plan_walk":
        if "turn_back_by" not in facts:
            return "Better skip the walk for now."
        park = facts.get("park")
        return f"A short walk to {park['name']} fits your free time." if park else "Take a short walk around your area."
    if intent == "other":
        return "I can only help plan walks and check weather and daylight."
    return "Here's what I found."


def _headline_ok(text: str, intent: str, facts: dict) -> bool:
    """Reject placeholders and invented numbers.

    Walk plans may not contain numbers at all (they belong in the lines); other
    replies may only use numbers and times that appear in FACTS.
    """
    if not text or "<" in text or ">" in text:
        return False
    if intent == "plan_walk":
        # With no park in the data the model invents one, so use the fixed headline.
        park = facts.get("park")
        return bool(park) and not re.search(r"\d", text) and park["name"].lower() in text.lower()
    known = json.dumps(facts)
    return all(num in known for num in re.findall(r"\d+(?:[.:]\d+)?", text))


@dataclass
class Reply:
    headline: str
    lines: list[str]
    templated: bool  # True when the model's headline failed checks and a fixed one was used

    @property
    def text(self) -> str:
        return "\n".join([self.headline, *self.lines])


def respond(result: LoopResult, responder=None) -> Reply:
    facts = build_facts(result)
    lines = detail_lines(result.intent, facts)
    if result.intent == "other":
        # Nothing to phrase; the model wrote a poem when asked to decline one.
        return Reply(fallback_headline("other", facts), lines, templated=False)
    responder = responder or make_responder()
    system = f"{COMMON_RULES}\n\n{ACTION_RULES.get(result.intent, ACTION_RULES['other'])}"
    reply = responder.invoke([SystemMessage(system), HumanMessage(build_context(result, facts))])
    headline = _SENTENCE_END.split(reply.content.strip())[0].strip().strip('"')

    if not _headline_ok(headline, result.intent, facts):
        return Reply(fallback_headline(result.intent, facts), lines, templated=True)
    return Reply(headline, lines, templated=False)
