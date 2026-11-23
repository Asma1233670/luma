"""Runtime settings, read from environment variables (see .env.example)."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    # Zone used to interpret times the user gives without an offset ("tomorrow 3pm").
    timezone: str = os.getenv("LUMA_TZ", "UTC")

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