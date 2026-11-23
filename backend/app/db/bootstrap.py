"""Create constraints, indexes and built-in types. Safe to run on every startup."""
from pathlib import Path

from .client import LumaDB
from .registry import seed_builtin_types

SCHEMA_FILE = Path(__file__).parent / "schema" / "bootstrap.cypher"
# Nodes that carry embeddings -> vector index names.
VECTOR_TARGETS = {"Detection": "detection_embedding", "Image": "image_embedding",
                  "TextBlock": "textblock_embedding", "Entity": "entity_embedding"}


def _statements(text: str) -> list[str]:
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("//")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def bootstrap(db: LumaDB, embed_dim: int) -> None:
    if not isinstance(embed_dim, int) or embed_dim <= 0:
        raise ValueError("embed_dim must be a positive int")
    for stmt in _statements(SCHEMA_FILE.read_text()):
        db.write(stmt)
    for label, index in VECTOR_TARGETS.items():  # f-string is safe: constants + validated int
        db.write(
            f"CREATE VECTOR INDEX {index} IF NOT EXISTS FOR (n:{label}) ON (n.embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {embed_dim}, `vector.similarity_function`: 'cosine'}}}}"
        )
    seed_builtin_types(db)