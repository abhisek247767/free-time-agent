"""Step 2: sunrise, sunset and golden hour via astral (fully offline)."""

from datetime import date as date_cls, datetime, tzinfo
from zoneinfo import ZoneInfo

from astral import Observer
from astral.sun import SunDirection, golden_hour, sun


def _local_tz() -> tzinfo:
    return datetime.now().astimezone().tzinfo


def get_sun_times(
    lat: float, lon: float, date: date_cls | None = None, tz: str | None = None
) -> dict:
    """Sunrise, sunset and morning/evening golden hour for a place and date.

    `tz` is an IANA name like "Asia/Kolkata"; defaults to the system timezone.
    """
    date = date or date_cls.today()
    tzinfo = ZoneInfo(tz) if tz else _local_tz()
    observer = Observer(latitude=lat, longitude=lon)

    s = sun(observer, date=date, tzinfo=tzinfo)
    morning = golden_hour(observer, date=date, direction=SunDirection.RISING, tzinfo=tzinfo)
    evening = golden_hour(observer, date=date, direction=SunDirection.SETTING, tzinfo=tzinfo)

    return {
        "date": date.isoformat(),
        "sunrise": s["sunrise"],
        "sunset": s["sunset"],
        "golden_hour_morning": (morning[0], morning[1]),
        "golden_hour_evening": (evening[0], evening[1]),
    }
