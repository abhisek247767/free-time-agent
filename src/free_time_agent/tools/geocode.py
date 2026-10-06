"""Step 3: place name -> lat/lon via Nominatim (OpenStreetMap).

Nominatim usage policy: identifying User-Agent, max 1 request/second, and
cache results. All three are handled here.
"""

import threading
import time

import httpx

from ._common import cache_path, env_float, env_str, http_client, read_json, write_json

NOMINATIM_URL = env_str("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
# Nominatim's policy allows at most 1 request/second; don't set this below 1.0.
MIN_INTERVAL_S = max(1.0, env_float("NOMINATIM_MIN_INTERVAL_S", 1.0))

_lock = threading.Lock()
_last_request = 0.0


def _rate_limited_get(params: dict) -> list[dict]:
    global _last_request
    with _lock:
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            with http_client() as client:
                resp = client.get(NOMINATIM_URL, params=params)
                resp.raise_for_status()
                return resp.json()
        finally:
            _last_request = time.monotonic()


def geocode(place_name: str) -> dict | None:
    """Return {"name", "lat", "lon"} for the best match, or None if not found."""
    key = place_name.strip().lower()
    path = cache_path("geocode.json")
    cache = read_json(path) or {}
    if key in cache:
        return cache[key]

    try:
        results = _rate_limited_get({"q": place_name, "format": "json", "limit": 1})
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Geocoding failed and '{place_name}' is not cached: {exc}") from exc

    if not results:
        return None
    top = results[0]
    hit = {"name": top["display_name"], "lat": float(top["lat"]), "lon": float(top["lon"])}
    cache[key] = hit
    write_json(path, cache)
    return hit
