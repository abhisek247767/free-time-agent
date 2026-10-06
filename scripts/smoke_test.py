"""Step 7: call every tool once and print the output. No LLM involved.

Run with:  uv run python scripts/smoke_test.py ["Place name"]
"""

import sqlite3
import sys
import traceback
from datetime import datetime
from pprint import pprint

from free_time_agent.tools._common import cache_path
from free_time_agent.tools import (
    find_green_spaces,
    geocode,
    get_sun_times,
    get_walk_route,
    get_weather,
    log_outing,
    outing_stats,
    turnaround_time,
    walk_time,
)

PLACE = sys.argv[1] if len(sys.argv) > 1 else "Hussain Sagar, Hyderabad"
FALLBACK = {"name": PLACE, "lat": 17.4239, "lon": 78.4738}

failures = []


def run(label, fn, *args, **kwargs):
    print(f"\n=== {label} ===")
    try:
        result = fn(*args, **kwargs)
        pprint(result, sort_dicts=False, width=100)
        return result
    except Exception:
        traceback.print_exc()
        failures.append(label)
        return None


loc = run("geocode", geocode, PLACE) or FALLBACK
lat, lon = loc["lat"], loc["lon"]

run("get_weather", get_weather, lat, lon, hours=4)
sun = run("get_sun_times", get_sun_times, lat, lon)
spaces = run("find_green_spaces", find_green_spaces, lat, lon, radius_m=1500, limit=5)

dest = (spaces or {}).get("places") or [{"lat": lat + 0.01, "lon": lon + 0.01}]
route = run("get_walk_route", get_walk_route, (lat, lon), (dest[-1]["lat"], dest[-1]["lon"]))

distance = (route or {}).get("distance_km", 2.0)
run("walk_time", walk_time, distance, "normal")

sunset = sun["sunset"] if sun else datetime.now().replace(hour=18, minute=0)
start = sunset.replace(hour=16, minute=30, second=0, microsecond=0)
run("turnaround_time", turnaround_time, start, sunset, buffer_min=15)

logged = run("log_outing", log_outing, PLACE, distance, "smoke test")
run("outing_stats", outing_stats)

# Remove the test row so it doesn't skew real walk stats.
if logged:
    with sqlite3.connect(cache_path("outings.db")) as conn:
        conn.execute("DELETE FROM outings WHERE id = ?", (logged["id"],))

print("\n" + ("All tools OK." if not failures else f"FAILED: {', '.join(failures)}"))
sys.exit(1 if failures else 0)
