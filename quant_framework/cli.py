"""CLI entry point for quant_framework."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import yaml


@click.group()
@click.version_option(package_name="quant_framework")
def cli() -> None:
    """quant_framework — quantitative research toolkit."""


@cli.command()
@click.option(
    "--persona",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to the persona YAML configuration file.",
)
@click.option(
    "--transport",
    default="sse",
    type=click.Choice(["sse", "stdio"], case_sensitive=False),
    show_default=True,
    help="MCP transport to use.",
)
def serve(persona: str, transport: str) -> None:
    """Load a persona config, initialise connectors, and start the MCP server."""
    persona_path = Path(persona)
    click.echo(f"Loading persona from {persona_path.name} ...")

    # Read persona to discover which connectors/functions to load
    with open(persona_path) as f:
        config = yaml.safe_load(f) or {}

    # ── 1. Ensure modelling functions are registered ─────────────────────
    click.echo("  Registering functions ...")
    # Import the functions module so @register_function decorators fire
    import quant_framework.functions.modelling  # noqa: F401
    from quant_framework.core import FunctionRegistry

    fn_names = config.get("functions", [])
    available = FunctionRegistry.list()
    for fn in fn_names:
        if fn not in available:
            click.secho(f"  ⚠  Function '{fn}' not found in registry — skipping", fg="yellow")
    click.echo(f"  ✓ {len(available)} function(s) available in registry")

    # ── 2. Initialise connectors ─────────────────────────────────────────
    connector_names = config.get("connectors", [])
    if connector_names:
        click.echo("  Initialising connectors ...")
        from quant_framework.connectors import ConnectorRegistry

        for cname in connector_names:
            conn_cls = ConnectorRegistry.get(cname)
            if conn_cls is None:
                click.secho(f"  ⚠  Connector '{cname}' not found in registry — skipping", fg="yellow")
                continue

            conn = conn_cls()
            # Auto-connect using environment variables (e.g. FRED_API_KEY)
            env_key = f"{cname.upper()}_API_KEY"
            api_key = os.getenv(env_key)
            if api_key:
                conn.connect({"api_key": api_key})
                click.echo(f"  ✓ {cname} connected (key from ${env_key})")
            else:
                click.secho(
                    f"  ⚠  {cname}: no ${env_key} found — connector not connected",
                    fg="yellow",
                )

    # ── 3. Generate and start MCP server ─────────────────────────────────
    from quant_framework.mcp.generator import MCPServerGenerator

    gen = MCPServerGenerator(persona_path)
    server = gen.generate()

    host = config.get("host", "127.0.0.1")
    port = config.get("port", 8000)
    name = config.get("name", "quant_framework")

    click.echo(f"\n🚀 Starting '{name}' MCP server on {host}:{port} ({transport}) ...\n")
    gen.serve(transport=transport)


def main() -> None:
    """Package-level entry point."""
    cli()


if __name__ == "__main__":
    main()
