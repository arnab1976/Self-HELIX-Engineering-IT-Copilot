"""Configuration: environment variables (secrets, endpoints) + configs/helix.yaml (platform policy)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HELIX_", env_file=ROOT / ".env", extra="ignore")

    env: str = "dev"
    data_dir: Path = ROOT / "data"
    state_dir: Path = ROOT / "var"
    config_file: Path = ROOT / "configs" / "helix.yaml"

    jwt_secret: str = "dev-only-helix-jwt-secret-change-me-0001"
    oidc_issuer: str = ""
    oidc_audience: str = "api://helix"
    approval_secret: str = "dev-only-helix-approval-secret-change-me"

    t1a_url: str = ""
    t1a_model: str = "qwen2.5-coder:1.5b-base"
    t1b_url: str = ""
    t1b_model: str = "qwen2.5-coder:7b-instruct"
    t1c_url: str = ""
    t1c_model: str = "llama3.1:8b-instruct"
    t2_url: str = ""
    t2_model: str = "llama3.1:8b-instruct"
    t3_url: str = ""
    t3_model: str = ""
    t3_api_key: str = ""

    opa_url: str = ""
    qdrant_url: str = ""
    embeddings_url: str = ""
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    github_token: str = ""
    servicenow_instance: str = ""
    servicenow_user: str = ""
    servicenow_password: str = ""
    confluence_site: str = ""
    confluence_email: str = ""
    confluence_api_token: str = ""
    confluence_space_id: str = ""
    loki_url: str = ""
    loki_query: str = ""
    # Employee Self-Service portal tickets (Docker postgres:16 · profile stores)
    postgres_url: str = "postgresql://postgres:helix@127.0.0.1:5432/helix"

    @property
    def is_dev(self) -> bool:
        return self.env.lower() == "dev"

    def path(self, *parts: str) -> Path:
        p = Path(self.state_dir).joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def tier_endpoint(self, tier: str) -> tuple[str, str, str]:
        t = tier.lower()
        return getattr(self, f"{t}_url", ""), getattr(self, f"{t}_model", ""), getattr(self, f"{t}_api_key", "")


@lru_cache
def settings() -> Settings:
    s = Settings()
    Path(s.state_dir).mkdir(parents=True, exist_ok=True)
    return s


@lru_cache
def cfg() -> dict[str, Any]:
    """Platform configuration from configs/helix.yaml."""
    with open(settings().config_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_yaml(rel: str) -> dict[str, Any]:
    with open(ROOT / rel, encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_class(repo: str | None) -> str | None:
    if not repo:
        return None
    return cfg()["repositories"].get(repo, "internal")
