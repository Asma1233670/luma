"""Per-run context handed to every tool via the Agents SDK (`ctx.context`)."""
from dataclasses import dataclass

from backend.app.db.client import LumaDB


@dataclass
class LumaContext:
    db: LumaDB