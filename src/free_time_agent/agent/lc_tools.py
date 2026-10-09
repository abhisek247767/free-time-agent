"""Step 8: the plain tools wrapped as LangChain tools.

Qwen fills in the arguments from these schemas and docstrings, so they stay
short and say when to call each tool. Outputs are small and JSON-friendly
(times as "HH:MM") because they go back to the model as context.
"""

from datetime import date as date_cls

from langchain_core.tools import tool

from free_time_agent import tools as base


def _hhmm(dt) -> str:
    return dt.strftime("%H:%M")


@tool
def geocode(place_name: str) -> dict:
    """Get lat/lon for a place name like "Gachibowli, Hyderabad". Call this first whenever the user names a place."""
    hit = base.geocode(place_name)
    if hit is None:
        return {"error": f"No match for {place_name!r}"}
    short_name = ", ".join(hit["name"].split(", ")[:3])
    return {"name": short_name, "lat": round(hit["lat"], 5), "lon": round(hit["lon"], 5)}


@tool
def get_weather(lat: float, lon: float, hours: int = 3) -> dict:
    """Hourly temperature, rain chance and wind for the next few hours at lat/lon."""
    w = base.get_weather(lat, lon, hours=hours)
    return {"stale": w["stale"], "hours": w["hours"]}


@tool
def get_sun_times(lat: float, lon: float, date: str = "") -> dict:
    """Sunrise, sunset and golden hour at lat/lon. date is YYYY-MM-DD, empty for today."""
    day = date_cls.fromisoformat(date) if date else None
    s = base.get_sun_times(lat, lon, day)
    return {
        "date": s["date"],
        "sunrise": _hhmm(s["sunrise"]),
        "sunset": _hhmm(s["sunset"]),
        "golden_hour_evening": f"{_hhmm(s['golden_hour_evening'][0])}-{_hhmm(s['golden_hour_evening'][1])}",
    }


@tool
def find_green_spaces(lat: float, lon: float, radius_m: int = 2000) -> dict:
    """Parks, gardens, woods and trails near lat/lon, nearest first."""
    result = base.find_green_spaces(lat, lon, radius_m=radius_m, limit=50)
    found = result["places"]
    # Named places are the only useful suggestions; keep unnamed ones as a fallback.
    named = [p for p in found if not p["name"].startswith("Unnamed")]
    top = (named or found)[:5]
    return {
        "source": result["source"],
        "places": [
            {k: p[k] for k in ("name", "kind", "lat", "lon", "distance_m")} for p in top
        ]
    }


@tool
def get_walk_route(from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> dict:
    """Walking distance and time between two points. Use after picking a park."""
    return base.get_walk_route((from_lat, from_lon), (to_lat, to_lon))


@tool
def walk_time(distance_km: float, pace: str = "normal") -> dict:
    """Minutes to walk distance_km. pace is "slow", "normal" or "brisk"."""
    return base.walk_time(distance_km, pace)


@tool
def turnaround_time(start_time: str, sunset: str, buffer_min: int = 15) -> dict:
    """When to turn back on an out-and-back walk to be home before sunset. Times are "HH:MM"."""
    r = base.turnaround_time(start_time, sunset, buffer_min)
    out = {k: v for k, v in r.items() if k not in ("turnaround_at", "back_by")}
    out["back_by"] = _hhmm(r["back_by"])
    if "turnaround_at" in r:
        out["turnaround_at"] = _hhmm(r["turnaround_at"])
    return out


ALL_TOOLS = [
    geocode,
    get_weather,
    get_sun_times,
    find_green_spaces,
    get_walk_route,
    walk_time,
    turnaround_time,
]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
