"""Function registry, decorator, and result type for quant_framework."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


# ── FunctionResult ───────────────────────────────────────────────────────────

@dataclass
class FunctionResult:
    """Wraps the output of a registered function with metrics and tracing."""

    output: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)


# ── FunctionRegistry ─────────────────────────────────────────────────────────

@dataclass
class _FunctionMeta:
    """Internal metadata stored for each registered function."""
    name: str
    category: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    fn: Callable[..., Any]


class FunctionRegistry:
    """Global registry that maps names to decorated Python functions."""

    _functions: Dict[str, _FunctionMeta] = {}

    @classmethod
    def register(cls, meta: _FunctionMeta) -> None:
        """Add a function to the registry."""
        cls._functions[meta.name] = meta

    @classmethod
    def get(cls, name: str) -> Optional[_FunctionMeta]:
        """Retrieve function metadata by name."""
        return cls._functions.get(name)

    @classmethod
    def list(cls) -> List[str]:
        """List all registered function names."""
        return list(cls._functions.keys())

    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """List function names filtered by category."""
        return [
            m.name for m in cls._functions.values() if m.category == category
        ]

    @classmethod
    def call(cls, name: str, *args: Any, **kwargs: Any) -> FunctionResult:
        """Look up a registered function by *name* and invoke it."""
        meta = cls._functions.get(name)
        if meta is None:
            raise KeyError(f"No function registered under '{name}'")
        return meta.fn(*args, **kwargs)


# ── @register_function decorator ─────────────────────────────────────────────

def register_function(
    name: str,
    category: str = "general",
    input_schema: Optional[Dict[str, Any]] = None,
    output_schema: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator that registers a Python function with metadata into
    the global :class:`FunctionRegistry`.

    Usage::

        @register_function(
            name="calc_rsi",
            category="technical",
            input_schema={"prices": "List[float]", "period": "int"},
            output_schema={"rsi": "float"},
        )
        def calc_rsi(prices, period=14):
            ...
            return FunctionResult(output={"rsi": value})

    Args:
        name: Unique name for the function.
        category: Logical grouping (e.g. ``"technical"``, ``"macro"``).
        input_schema: Dict describing expected inputs.
        output_schema: Dict describing expected outputs.
    """

    def decorator(fn: F) -> F:
        meta = _FunctionMeta(
            name=name,
            category=category,
            input_schema=input_schema or {},
            output_schema=output_schema or {},
            fn=fn,
        )
        FunctionRegistry.register(meta)
        return fn

    return decorator
