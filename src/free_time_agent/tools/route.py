"""Step 5: walking distance/duration between two points.

Order of attempts:
1. OpenRouteService, if ORS_API_KEY is set (free signup, generous daily limit).
2. OSRM foot profile on routing.openstreetmap.de (no key, light use only).
   The router.project-osrm.org demo only serves the car profile, so it would
   return driving distances.
3. Offline: straight-line haversine distance x 1.3.
"""

import os

import httpx

from ._common import env_float, env_str, haversine_km, http_client

Point = tuple[float, float]  # (lat, lon)

ORS_URL = env_str("ORS_URL", "https://api.openrouteservice.org/v2/directions/foot-walking")
OSRM_FOOT_URL = env_str("OSRM_FOOT_URL", "https://routing.openstreetmap.de/routed-foot/route/v1/foot")
DETOUR_FACTOR = env_float("ROUTE_DETOUR_FACTOR", 1.3)


def _ors(start: Point, end: Point, api_key: str) -> dict:
    body = {"coordinates": [[start[1], start[0]], [end[1], end[0]]]}
    with http_client() as client:
        resp = client.post(ORS_URL, json=body, headers={"Authorization": api_key})
        resp.raise_for_status()
        summary = resp.json()["routes"][0]["summary"]
    return {"distance_km": summary["distance"] / 1000, "duration_min": summary["duration"] / 60}


def _osrm(start: Point, end: Point) -> dict:
    url = f"{OSRM_FOOT_URL}/{start[1]},{start[0]};{end[1]},{end[0]}"
    with http_client() as client:
        resp = client.get(url, params={"overview": "false"})
        resp.raise_for_status()
        route = resp.json()["routes"][0]
    return {"distance_km": route["distance"] / 1000, "duration_min": route["duration"] / 60}


def get_walk_route(start: Point, end: Point) -> dict:
    """Walking distance (km) and, when routed online, duration (min)."""
    providers = []
    if api_key := os.environ.get("ORS_API_KEY"):
        providers.append(("openrouteservice", lambda: _ors(start, end, api_key)))
    providers.append(("osrm", lambda: _osrm(start, end)))

    for name, call in providers:
        try:
            result = call()
            return {
                "source": name,
                "distance_km": round(result["distance_km"], 2),
                "duration_min": round(result["duration_min"]),
            }
        except (httpx.HTTPError, KeyError, IndexError):
            continue

    straight = haversine_km(*start, *end)
    return {
        "source": "haversine-estimate",
        "distance_km": round(straight * DETOUR_FACTOR, 2),
        "duration_min": None,
    }
