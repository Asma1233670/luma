"""Validation and normalisation for names the agent wants to put into the graph.

Cypher cannot parameterise labels or relationship types, so any name that ends up
inside a query string MUST pass through these functions first.
"""
import re

LABEL_RE = re.compile(r"^[A-Z][A-Za-z0-9]{1,39}$")      # PascalCase
REL_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,39}$")      # UPPER_SNAKE

# Labels in the media / perception layers (and the registry itself) are owned by
# Luma's pipelines. The agent may read them but never create or relabel them.
LOCKED_LABELS = frozenset({
    "Entity", "Event", "EntityType",
    "Session", "Image", "AudioClip", "TextBlock",
    "Detection", "Utterance",
})


class InvalidName(ValueError):
    pass


def normalize_label(raw: str) -> str:
    """'eye glasses' / 'eye_glasses' / 'EYE-GLASSES' -> 'EyeGlasses'."""
    parts = re.split(r"[^A-Za-z0-9]+", raw.strip())
    label = "".join(p[:1].upper() + p[1:] for p in parts if p)
    return validate_label(label)


def normalize_rel_type(raw: str) -> str:
    """'last seen in' / 'lastSeenIn' -> 'LAST_SEEN_IN'."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw.strip())
    parts = re.split(r"[^A-Za-z0-9]+", spaced)
    return validate_rel_type("_".join(p.upper() for p in parts if p))


def validate_label(label: str) -> str:
    if not LABEL_RE.match(label):
        raise InvalidName(f"Invalid node label {label!r}: use PascalCase letters/digits, 2-40 chars.")
    return label


def validate_rel_type(rel: str) -> str:
    if not REL_TYPE_RE.match(rel):
        raise InvalidName(f"Invalid relationship type {rel!r}: use UPPER_SNAKE_CASE, 2-40 chars.")
    return rel


def assert_agent_may_use_label(label: str) -> None:
    if label in LOCKED_LABELS:
        raise InvalidName(f"{label!r} is a locked label owned by Luma's pipelines.")