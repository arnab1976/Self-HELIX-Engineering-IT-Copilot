"""Test session: isolated state dir, synthetic data, bootstrapped models and index (about 10 seconds)."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATE = Path(tempfile.mkdtemp(prefix="helix-test-"))
os.environ["HELIX_STATE_DIR"] = str(STATE)
os.environ["HELIX_ENV"] = "dev"
for t in ("T1A", "T1B", "T1C", "T2", "T3"):
    os.environ[f"HELIX_{t}_URL"] = ""          # force the deterministic mock providers
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def bootstrapped():
    if not (ROOT / "data" / "generated" / "tickets.csv").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_sample_data.py")], check=True)
    import scripts.bootstrap as b  # noqa: F401  (import after env is set)

    assert b.main() == 0
    yield STATE


@pytest.fixture(scope="session")
def client(bootstrapped):
    from fastapi.testclient import TestClient

    from nexus.api.main import app

    return TestClient(app)


@pytest.fixture(scope="session")
def auth(client):
    cache = {}

    def _h(user: str) -> dict:
        if user not in cache:
            cache[user] = {"Authorization": "Bearer " + client.post("/v1/dev/token", json={"user": user}).json()["access_token"]}
        return cache[user]
    return _h
