"""Step 1: weather forecast from Open-Meteo (free, no key), cached for offline use."""

from datetime import datetime, timedelta, timezone

import httpx

from ._common import cache_path, coord_key, http_client, read_json, write_json

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_FIELDS = "temperature_2m,precipitation_probability,wind_speed_10m"


def _fetch(lat: float, lon: float) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_FIELDS,
        "timezone": "auto",
        "forecast_days": 2,
    }
    with http_client() as client:
        resp = client.get(OPEN_METEO_URL, params=params)
        resp.raise_for_status()
        return resp.json()


def _next_hours(raw: dict, hours: int) -> list[dict]:
    hourly = raw["hourly"]
    # Open-Meteo returns local times without an offset when timezone=auto.
    local_now = datetime.now(timezone.utc) + timedelta(seconds=raw.get("utc_offset_seconds", 0))
    current_hour = local_now.replace(tzinfo=None, minute=0, second=0, microsecond=0)

    rows = []
    for i, t in enumerate(hourly["time"]):
        if datetime.fromisoformat(t) < current_hour:
            continue
        rows.append(
            {
                "time": t,
                "temperature_c": hourly["temperature_2m"][i],
                "rain_chance_pct": hourly["precipitation_probability"][i],
                "wind_kmh": hourly["wind_speed_10m"][i],
            }
        )
        if len(rows) == hours:
            break
    return rows


def get_weather(lat: float, lon: float, hours: int = 6) -> dict:
    """Temperature, rain chance and wind for the next `hours` hours.

    Every successful response is saved to data/weather/<lat>_<lon>.json; if the
    network call fails, the last saved response is used and `stale` is True.
    """
    path = cache_path("weather", f"{coord_key(lat, lon)}.json")
    stale = False
    try:
        raw = _fetch(lat, lon)
        raw["_fetched_at"] = datetime.now(timezone.utc).isoformat()
        write_json(path, raw)
    except httpx.HTTPError as exc:
        raw = read_json(path)
        if raw is None:
            raise RuntimeError(f"Weather unavailable and no cached data: {exc}") from exc
        stale = True

    return {
        "timezone": raw.get("timezone"),
        "fetched_at": raw.get("_fetched_at"),
        "stale": stale,
        "hours": _next_hours(raw, hours),
    }
