"""Authentication and authorisation.

Production: OIDC bearer tokens from Okta / Entra ID, verified against the issuer's JWKS (RS256).
Dev (HELIX_ENV=dev): HS256 tokens minted by POST /v1/dev/token for the personas in configs/helix.yaml.
Roles map to readable repository classes (RBAC); the policy gate uses them as hard inputs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from common.config import cfg, settings


@dataclass
class Principal:
    sub: str
    name: str
    roles: list[str]
    manager: str | None = None
    device: str | None = None
    groups: list[str] = field(default_factory=list)

    def has_role(self, role: str) -> bool:
        return role in self.roles

    @property
    def repo_classes(self) -> set[str]:
        out: set[str] = set()
        for r in self.roles:
            out.update(cfg()["roles"].get(r, {}).get("repo_classes", []))
        return out

    def can_read(self, repo_class: str | None) -> bool:
        return repo_class is None or repo_class in self.repo_classes


def mint_dev_token(user: str, ttl_s: int = 8 * 3600) -> str:
    users = cfg()["dev_users"]
    if user not in users:
        raise KeyError(user)
    u = users[user]
    now = int(time.time())
    claims = {"sub": user, "name": u["name"], "roles": u["roles"], "manager": u.get("manager"), "device": u.get("device"),
              "iat": now, "exp": now + ttl_s, "iss": "helix-dev", "aud": settings().oidc_audience}
    return jwt.encode(claims, settings().jwt_secret, algorithm="HS256")


@lru_cache
def _jwks_client():
    return jwt.PyJWKClient(settings().oidc_issuer.rstrip("/") + "/.well-known/jwks.json") if settings().oidc_issuer else None


def decode(token: str) -> Principal:
    s = settings()
    try:
        if s.oidc_issuer:
            key = _jwks_client().get_signing_key_from_jwt(token).key
            c = jwt.decode(token, key, algorithms=["RS256"], audience=s.oidc_audience, issuer=s.oidc_issuer)
            roles = c.get("roles") or c.get("groups") or ["employee"]
        else:
            c = jwt.decode(token, s.jwt_secret, algorithms=["HS256"], audience=s.oidc_audience)
            roles = c.get("roles", ["employee"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid token: {exc}") from exc
    return Principal(c["sub"], c.get("name", c["sub"]), list(roles), c.get("manager"), c.get("device"), c.get("groups", []))


_bearer = HTTPBearer(auto_error=False)


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> Principal:
    if not creds:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token (dev: POST /v1/dev/token)")
    return decode(creds.credentials)


def require_role(role: str):
    def dep(user: Principal = Depends(current_user)) -> Principal:
        if not user.has_role(role):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"role '{role}' required")
        return user
    return dep
