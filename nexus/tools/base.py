"""Tool runtime (spec 7.4): typed tools with timeouts, retry budgets, circuit breakers, idempotency keys and
approval tokens bound to the exact action and parameters the approver saw.

Every tool here is also exposed over MCP by nexus/tools/mcp_server.py.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout
from dataclasses import dataclass, field
from typing import Any

from common.config import cfg, settings
from common.store import store
from common.tracing import current_trace_id, span


class ToolError(Exception):
    pass


@dataclass
class ToolSpec:
    name: str
    system: str
    fn: Callable[..., Any]
    write: bool = False
    approval: str | None = None          # None | step_up | manager | sre | author
    timeout_s: float = 5.0
    retries: int = 1                     # reads only; writes are never retried blindly
    description: str = ""
    params: dict = field(default_factory=dict)


class CircuitBreaker:
    def __init__(self, failures: int = 5, reset_s: float = 30.0):
        self.max, self.reset_s = failures, reset_s
        self.count, self.opened = 0, 0.0

    def allow(self) -> bool:
        return self.count < self.max or time.time() - self.opened > self.reset_s

    def record(self, ok: bool) -> None:
        if ok:
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.max:
                self.opened = time.time()


# ------------------------------------------------------------------------------------------- approvals
class Approvals:
    """token = exp~approver~action_id~HMAC_SHA256(secret, canonical{action_id, tool, params, approver, exp})"""

    @staticmethod
    def _canon(action_id: str, tool: str, params: dict, approver: str, exp: int) -> bytes:
        return json.dumps({"a": action_id, "t": tool, "p": params, "u": approver, "e": exp}, sort_keys=True, separators=(",", ":")).encode()

    def mint(self, action_id: str, tool: str, params: dict, approver: str) -> str:
        exp = int(time.time()) + 60 * cfg()["approvals"]["token_ttl_minutes"]
        sig = hmac.new(settings().approval_secret.encode(), self._canon(action_id, tool, params, approver, exp), hashlib.sha256).hexdigest()
        return f"{exp}~{approver}~{action_id}~{sig}"

    def verify(self, token: str, tool: str, params: dict) -> tuple[bool, str]:
        try:
            exp_s, approver, action_id, sig = token.split("~", 3)
            exp = int(exp_s)
        except ValueError:
            return False, "malformed token"
        if time.time() > exp:
            return False, "token expired"
        good = hmac.new(settings().approval_secret.encode(), self._canon(action_id, tool, params, approver, exp), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(good, sig):
            return False, "token does not match this tool and these parameters"
        return True, approver


approvals = Approvals()


# ------------------------------------------------------------------------------------------- registry
class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, ToolSpec] = {}
        self.breakers: dict[str, CircuitBreaker] = {}
        self.pool = ThreadPoolExecutor(max_workers=8)
        self._lock = threading.Lock()

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec
        self.breakers[spec.name] = CircuitBreaker()

    def tool(self, name: str, system: str, write: bool = False, approval: str | None = None, timeout_s: float = 5.0,
             description: str = "", params: dict | None = None):
        def deco(fn):
            self.register(ToolSpec(name, system, fn, write, approval, timeout_s, 0 if write else 1, description or (fn.__doc__ or "").strip(),
                                   params or {}))
            return fn
        return deco

    def catalogue(self) -> list[dict]:
        return [{"name": t.name, "system": t.system, "write": t.write, "approval": t.approval, "timeout_s": t.timeout_s,
                 "description": t.description, "params": t.params} for t in self.tools.values()]

    def call(self, name: str, params: dict | None = None, actor: str = "system", approval_token: str | None = None,
             idempotency_key: str | None = None, allowed: set[str] | None = None) -> Any:
        params = params or {}
        spec = self.tools.get(name)
        if spec is None:
            raise ToolError(f"unknown tool {name}")
        if allowed is not None and name not in allowed:
            raise ToolError(f"tool {name} is not on this workflow step's allow-list")
        if spec.write and spec.approval in ("sre", "manager", "author"):
            if not approval_token:
                raise ToolError(f"{name} requires an approval token ({spec.approval})")
            ok, who = approvals.verify(approval_token, name, params)
            if not ok:
                raise ToolError(f"approval rejected: {who}")
        if idempotency_key:
            prev = store().one("SELECT result FROM idempotency WHERE key=?", (idempotency_key,))
            if prev:
                return json.loads(prev["result"])
        br = self.breakers[name]
        if not br.allow():
            raise ToolError(f"circuit open for {name}")
        attempts = 1 + spec.retries
        last: Exception | None = None
        for _ in range(attempts):
            with span("tool.call", tool=name, write=spec.write):
                fut = self.pool.submit(spec.fn, **params)
                try:
                    res = fut.result(timeout=spec.timeout_s)
                    br.record(True)
                    break
                except FutTimeout:
                    last = ToolError(f"{name} timed out after {spec.timeout_s}s")
                    br.record(False)
                except ToolError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    br.record(False)
        else:
            raise ToolError(f"{name} failed: {last!r}")
        if idempotency_key:
            store().execute("INSERT OR REPLACE INTO idempotency VALUES (?,?,?)", (idempotency_key, json.dumps(res, default=str), time.time()))
        if spec.write:
            store().audit(current_trace_id(), actor, f"tool:{name}", {"params": params, "result": res})
        return res


registry = ToolRegistry()
