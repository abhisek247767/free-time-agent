"""Shared helpers: data directory, JSON cache, HTTP client, distance math."""

import json
import math
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

# Settings come from the environment, optionally loaded from a .env file in the
# project root (found by searching up from this file). Real environment
# variables take precedence over .env.
load_dotenv()


def env_str(name: str, default: str) -> str:
    return os.environ.get(name) or default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


DATA_DIR = Path(env_str("FREE_TIME_AGENT_DATA", str(Path.cwd() / "data")))

# OSM services (Nominatim, Overpass) require an identifying User-Agent.
USER_AGENT = env_str("FREE_TIME_AGENT_USER_AGENT", "free-time-agent/0.1 (personal hobby project)")

HTTP_TIMEOUT = env_float("HTTP_TIMEOUT_S", 20.0)


def http_client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT)


def cache_path(*parts: str) -> Path:
    path = DATA_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def coord_key(lat: float, lon: float, precision: int = 2) -> str:
    """Round coordinates into a cache key (2 decimals is roughly 1 km)."""
    return f"{lat:.{precision}f}_{lon:.{precision}f}"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
