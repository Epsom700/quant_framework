"""MCP Server generator that auto-wires registered functions as MCP Tools."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from mcp.server.fastmcp import FastMCP

from quant_framework.core.function import FunctionRegistry, FunctionResult

logger = logging.getLogger(__name__)


class MCPServerGenerator:
    """Generate an MCP server from a persona YAML configuration.

    The persona YAML should follow this structure::

        name: "Quant Research Agent"
        description: "MCP server for quantitative research"     # optional
        host: "127.0.0.1"                                       # optional, default 127.0.0.1
        port: 8000                                               # optional, default 8000

        functions:
          - run_linear
          - run_random_forest
          - run_xgboost

        connectors:
          - fred

    ``generate()`` creates a :class:`mcp.server.fastmcp.FastMCP` instance and
    registers every listed function from the :class:`FunctionRegistry` as an
    MCP Tool.  ``serve()`` starts the SSE transport.
    """

    def __init__(self, persona_path: str | Path) -> None:
        self._persona_path = Path(persona_path)
        self._config: Dict[str, Any] = {}
        self._server: Optional[FastMCP] = None
        self._load_config()

    # ── config ───────────────────────────────────────────────────────────

    def _load_config(self) -> None:
        if not self._persona_path.exists():
            raise FileNotFoundError(
                f"Persona config not found: {self._persona_path}"
            )
        with open(self._persona_path, "r") as f:
            self._config = yaml.safe_load(f) or {}

    @property
    def name(self) -> str:
        return self._config.get("name", "quant_framework")

    @property
    def description(self) -> str:
        return self._config.get("description", "")

    @property
    def host(self) -> str:
        return self._config.get("host", "127.0.0.1")

    @property
    def port(self) -> int:
        return self._config.get("port", 8000)

    @property
    def function_names(self) -> List[str]:
        return self._config.get("functions", [])

    @property
    def connector_names(self) -> List[str]:
        return self._config.get("connectors", [])

    # ── generation ───────────────────────────────────────────────────────

    def generate(self) -> FastMCP:
        """Create a :class:`FastMCP` server and register Tools from the FunctionRegistry.

        Returns:
            The configured ``FastMCP`` instance (also stored as ``self.server``).
        """
        self._server = FastMCP(
            name=self.name,
            host=self.host,
            port=self.port,
        )

        registered = 0
        for fn_name in self.function_names:
            meta = FunctionRegistry.get(fn_name)
            if meta is None:
                logger.warning(
                    "Function '%s' listed in persona but not found in FunctionRegistry — skipping.",
                    fn_name,
                )
                continue

            self._register_tool(fn_name, meta)
            registered += 1

        logger.info(
            "Generated MCP server '%s' with %d tools on %s:%d",
            self.name, registered, self.host, self.port,
        )
        return self._server

    def _register_tool(self, fn_name: str, meta: Any) -> None:
        """Wrap a registered function as an MCP Tool on the server."""
        fn = meta.fn
        doc = (fn.__doc__ or "").strip()
        input_schema = meta.input_schema

        # Build the tool handler that converts FunctionResult → dict for MCP
        async def _tool_handler(**kwargs: Any) -> str:
            result = fn(**kwargs)
            if isinstance(result, FunctionResult):
                return json.dumps({
                    "output": _serialise(result.output),
                    "metrics": result.metrics,
                    "trace_id": result.trace_id,
                })
            return json.dumps(_serialise(result))

        # Give the handler the correct name so FastMCP labels it properly
        _tool_handler.__name__ = fn_name
        _tool_handler.__doc__ = doc
        _tool_handler.__qualname__ = fn_name

        # Register with FastMCP using decorator-style call
        self._server.tool(
            name=fn_name,
            description=doc or fn_name,
        )(_tool_handler)

    # ── serving ──────────────────────────────────────────────────────────

    @property
    def server(self) -> FastMCP:
        if self._server is None:
            raise RuntimeError("Server not generated. Call generate() first.")
        return self._server

    def serve(self, transport: str = "sse") -> None:
        """Start the MCP server.

        Args:
            transport: ``'sse'`` (default) or ``'stdio'``.
        """
        self.server.run(transport=transport)


# ── helpers ──────────────────────────────────────────────────────────────────

def _serialise(obj: Any) -> Any:
    """Make an object JSON-serialisable (best-effort)."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(i) for i in obj]
    if hasattr(obj, "tolist"):          # numpy arrays / pandas objects
        return obj.tolist()
    if hasattr(obj, "__dict__"):         # sklearn models → class name string
        return f"<{type(obj).__name__}>"
    return obj
