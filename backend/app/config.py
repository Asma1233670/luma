"""Runtime settings, read from environment variables (see .env.example)."""
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

load_dotenv()

TZ_FILE = Path("/etc/timezone")
LOCALTIME_FILE = Path("/etc/localtime")
ZONEINFO_DIR = Path("/usr/share/zoneinfo")


def _known_zone(name: str) -> str | None:
    """An IANA name Python can actually load, or None."""
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return None
    return name


def detect_timezone() -> str:
    """The user's timezone: explicit override, then the machine's setting.

    An IANA name (e.g. Africa/Tunis) is used rather than a fixed offset so that
    daylight-saving rules still apply to times like "tomorrow at 3pm".
    """
    for name in (os.getenv("LUMA_TZ"), os.getenv("TZ")):
        if name and (zone := _known_zone(name.lstrip(":"))):
            return zone

    # Debian/WSL normally record the zone in one of these two places.
    if TZ_FILE.is_file():
        if zone := _known_zone(TZ_FILE.read_text(encoding="utf-8").strip()):
            return zone
    if LOCALTIME_FILE.is_symlink():
        try:
            name = LOCALTIME_FILE.resolve().relative_to(ZONEINFO_DIR).as_posix()
        except ValueError:  # symlink points somewhere else
            name = ""
        if zone := _known_zone(name):
            return zone

    return "UTC"  # no usable zone information: at least be consistent


@dataclass(frozen=True)
class Settings:
    # Zone used to interpret times the user gives without an offset ("tomorrow 3pm").
    timezone: str = field(default_factory=detect_timezone)

    api_key=os.getenv("OPENAI_API_KEY")
    base_url=os.getenv(
        "BASE_URL",
        "https://promptix.tn/api/v1",
    )
    model=os.getenv("MODEL")
    db_path: str = os.getenv("LUMA_DB_PATH", "data/luma.relate")
    # Must match the embedding model you pick for vision/text. Changing it later
    # means recreating the vector indexes.
    embed_dim: int = int(os.getenv("LUMA_EMBED_DIM", "512"))


settings = Settings()
