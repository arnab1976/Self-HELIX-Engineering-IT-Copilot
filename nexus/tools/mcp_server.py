"""Expose the HELIX tool catalogue as an MCP server (Model Context Protocol) so any MCP-capable agent or IDE can
call the same governed tools. Read tools are exposed directly; write tools require an `approval_token`.

    pip install "mcp>=1.2"
    python -m nexus.tools.mcp_server              # stdio transport (Claude Desktop, IDEs)
"""
from __future__ import annotations

import json


def build_server():
    from mcp.server.fastmcp import FastMCP  # type: ignore

    from nexus.tools import registry

    mcp = FastMCP("helix-tools")

    def make(name: str):
        spec = registry.tools[name]

        def handler(params_json: str = "{}", approval_token: str = "", idempotency_key: str = "") -> str:
            params = json.loads(params_json or "{}")
            res = registry.call(name, params, actor="mcp", approval_token=approval_token or None,
                                idempotency_key=idempotency_key or None)
            return json.dumps(res, default=str)

        handler.__name__ = name.replace(".", "_")
        handler.__doc__ = f"[{spec.system}] {spec.description} (write={spec.write}, approval={spec.approval}). Pass params as JSON."
        return handler

    for n in registry.tools:
        mcp.tool(name=n.replace(".", "_"))(make(n))
    return mcp


if __name__ == "__main__":
    build_server().run()
