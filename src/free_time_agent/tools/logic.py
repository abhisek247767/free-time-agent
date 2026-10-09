"""Step 6: pure-logic tools — no network."""

from datetime import datetime, timedelta

PACES_MIN_PER_KM = {"slow": 15.0, "normal": 12.0, "brisk": 10.0}


def walk_time(distance_km: float, pace: float | str = "normal") -> dict:
    """Estimated walking duration. `pace` is min/km or one of slow/normal/brisk."""
    if isinstance(pace, str):
        if pace not in PACES_MIN_PER_KM:
            raise ValueError(f"pace must be a number or one of {list(PACES_MIN_PER_KM)}")
        pace = PACES_MIN_PER_KM[pace]
    minutes = distance_km * pace
    return {"distance_km": distance_km, "pace_min_per_km": pace, "duration_min": round(minutes)}


def _as_datetime(value: datetime | str, ref: datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        return value
    hh, mm = map(int, value.split(":"))
    base = ref or datetime.now()
    return base.replace(hour=hh, minute=mm, second=0, microsecond=0)


def turnaround_time(
    start_time: datetime | str, sunset: datetime | str, buffer_min: int = 15
) -> dict:
    """When to head back on an out-and-back walk to be home `buffer_min` before sunset.

    Accepts datetimes or "HH:MM" strings (interpreted as today). The outbound
    leg may use at most half the time between start and the safe deadline.
    """
    sunset_dt = _as_datetime(sunset)
    start_dt = _as_datetime(start_time, ref=sunset_dt if isinstance(sunset, datetime) else None)
    if start_dt.tzinfo is None and sunset_dt.tzinfo is not None:
        start_dt = start_dt.replace(tzinfo=sunset_dt.tzinfo)

    deadline = sunset_dt - timedelta(minutes=buffer_min)
    available = (deadline - start_dt).total_seconds() / 60
    if available <= 0:
        return {
            "safe": False,
            "message": "Too close to sunset — not enough daylight for a walk.",
            "back_by": deadline,
            "available_min": round(available),
        }

    turnaround = start_dt + timedelta(minutes=available / 2)
    return {
        "safe": True,
        "turnaround_at": turnaround,
        "back_by": deadline,
        "available_min": round(available),
        "max_outbound_min": round(available / 2),
    }
