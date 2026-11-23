import pytest

from app.db.bootstrap import bootstrap
from app.db.client import LumaDB

DIM = 4


'''@pytest.fixture(autouse=True)
def never_call_the_real_openai_api(monkeypatch):
    """Tests use fakes. If one accidentally reaches the real API (a wrong model wiring, say), it
    must fail fast and free instead of quietly spending money and passing or failing at random."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9")  # nothing listens here'''


@pytest.fixture
def db(tmp_path):
    d = LumaDB(str(tmp_path / "luma.relate"))
    bootstrap(d, embed_dim=DIM)
    yield d
    d.close()