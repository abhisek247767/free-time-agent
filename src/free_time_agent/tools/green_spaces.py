"""Step 4: nearby parks, gardens, woods and paths via the Overpass API.

Each query result is cached under data/green_spaces/. When offline, every
cached result is merged and filtered by distance, so once you have queried
your city the tool keeps working without a network.
"""

import httpx

from ._common import (
    cache_path,
    coord_key,
    env_float,
    env_str,
    haversine_km,
    http_client,
    read_json,
    write_json,
)

OVERPASS_URL = env_str("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
OVERPASS_TIMEOUT_S = env_float("OVERPASS_TIMEOUT_S", 40.0)

# Unnamed highway=path segments number in the thousands in any city and are
# useless to suggest, so only named paths are requested.
QUERY_TEMPLATE = """
[out:json][timeout:25];
(
  nwr["leisure"="park"](around:{r},{lat},{lon});
  nwr["leisure"="garden"](around:{r},{lat},{lon});
  nwr["natural"="wood"](around:{r},{lat},{lon});
  way["highway"="path"]["name"](around:{r},{lat},{lon});
);
out center tags;
"""


def _kind(tags: dict) -> str:
    if tags.get("leisure") in ("park", "garden"):
        return tags["leisure"]
    if tags.get("natural") == "wood":
        return "wood"
    return "path"


def _simplify(element: dict) -> dict | None:
    tags = element.get("tags", {})
    center = element.get("center") or element
    if "lat" not in center:
        return None
    return {
        "osm_id": f"{element['type']}/{element['id']}",
        "name": tags.get("name") or f"Unnamed {_kind(tags)}",
        "kind": _kind(tags),
        "lat": center["lat"],
        "lon": center["lon"],
    }


def _fetch(lat: float, lon: float, radius_m: int) -> list[dict]:
    query = QUERY_TEMPLATE.format(r=radius_m, lat=lat, lon=lon)
    with http_client() as client:
        resp = client.post(OVERPASS_URL, data={"data": query}, timeout=OVERPASS_TIMEOUT_S)
        resp.raise_for_status()
        elements = resp.json()["elements"]
    return [s for e in elements if (s := _simplify(e))]


def _all_cached() -> list[dict]:
    merged: dict[str, dict] = {}
    for path in cache_path("green_spaces", "x").parent.glob("*.json"):
        for place in read_json(path) or []:
            merged[place["osm_id"]] = place
    return list(merged.values())


def find_green_spaces(lat: float, lon: float, radius_m: int = 2000, limit: int = 20) -> dict:
    """Green spaces within `radius_m`, nearest first."""
    path = cache_path("green_spaces", f"{coord_key(lat, lon)}_{radius_m}.json")
    source = "network"
    places = read_json(path)
    if places is not None:
        source = "cache"
    else:
        try:
            places = _fetch(lat, lon, radius_m)
            write_json(path, places)
        except httpx.HTTPError:
            places = _all_cached()
            source = "offline-cache"

    results = []
    for p in places:
        dist_m = haversine_km(lat, lon, p["lat"], p["lon"]) * 1000
        if dist_m <= radius_m:
            results.append({**p, "distance_m": round(dist_m)})
    results.sort(key=lambda p: p["distance_m"])
    return {"source": source, "count": len(results), "places": results[:limit]}
