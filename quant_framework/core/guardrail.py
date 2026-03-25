"""Guardrail engine for validating function outputs against YAML-defined rules."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ── Exception ────────────────────────────────────────────────────────────────

class GuardrailViolation(Exception):
    """Raised when a function result violates a guardrail rule."""

    def __init__(
        self,
        function_name: str,
        rule: str,
        detail: str,
        role: Optional[str] = None,
    ) -> None:
        self.function_name = function_name
        self.rule = rule
        self.detail = detail
        self.role = role
        msg = f"[{function_name}] Guardrail violation ({rule}): {detail}"
        if role:
            msg += f" [role={role}]"
        super().__init__(msg)


# ── GuardrailEngine ──────────────────────────────────────────────────────────

class GuardrailEngine:
    """Load guardrail rules from a YAML file and validate function results.

    The YAML file should have the following structure::

        rules:
          <function_name>:
            max_records: 1000
            required_fields:
              - field_a
              - field_b
            redacted_fields:
              - secret_field
            roles:                    # optional per-role overrides
              analyst:
                max_records: 500
                redacted_fields:
                  - pii_field

        defaults:                     # optional fallback for unlisted functions
          max_records: 5000
          required_fields: []
          redacted_fields: []

    Supports **hot-reload**: the YAML file is automatically re-read whenever
    its modification time changes, so edits take effect without restarting.
    """

    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._last_mtime: float = 0.0
        self._load()

    # ── config loading / hot-reload ──────────────────────────────────────

    def _load(self) -> None:
        """Read and parse the YAML config file."""
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Guardrail config not found: {self._config_path}"
            )
        self._last_mtime = os.path.getmtime(self._config_path)
        with open(self._config_path, "r") as f:
            self._config = yaml.safe_load(f) or {}

    def _maybe_reload(self) -> None:
        """Re-read the file if its mtime has changed (hot-reload)."""
        try:
            current_mtime = os.path.getmtime(self._config_path)
        except OSError:
            return  # file disappeared – keep last known config
        if current_mtime != self._last_mtime:
            self._load()

    # ── rule resolution ──────────────────────────────────────────────────

    def _resolve_rules(
        self,
        function_name: str,
        role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Merge defaults → function rules → role overrides."""
        self._maybe_reload()

        defaults = self._config.get("defaults", {})
        fn_rules = self._config.get("rules", {}).get(function_name, {})

        # Start from defaults, overlay function-level rules
        merged: Dict[str, Any] = {**defaults, **fn_rules}

        # Apply per-role overrides if present
        if role and "roles" in fn_rules:
            role_overrides = fn_rules["roles"].get(role, {})
            merged.update(role_overrides)

        # Remove the nested 'roles' key from the merged result
        merged.pop("roles", None)
        return merged

    # ── validation ───────────────────────────────────────────────────────

    def validate(
        self,
        function_name: str,
        result: Dict[str, Any],
        role: Optional[str] = None,
    ) -> None:
        """Validate *result* against the guardrail rules for *function_name*.

        Args:
            function_name: The registered function name.
            result: The output dict from a ``FunctionResult.output``.
            role: Optional user role for per-role rule overrides.

        Raises:
            GuardrailViolation: If any rule is violated.
        """
        rules = self._resolve_rules(function_name, role)
        if not rules:
            return  # no rules defined — allow everything

        self._check_max_records(function_name, result, rules, role)
        self._check_required_fields(function_name, result, rules, role)
        self._check_redacted_fields(function_name, result, rules, role)

    # ── individual checks ────────────────────────────────────────────────

    @staticmethod
    def _check_max_records(
        fn: str,
        result: Dict[str, Any],
        rules: Dict[str, Any],
        role: Optional[str],
    ) -> None:
        max_records = rules.get("max_records")
        if max_records is None:
            return
        for key, value in result.items():
            if isinstance(value, list) and len(value) > max_records:
                raise GuardrailViolation(
                    function_name=fn,
                    rule="max_records",
                    detail=(
                        f"Field '{key}' has {len(value)} records, "
                        f"exceeds limit of {max_records}"
                    ),
                    role=role,
                )

    @staticmethod
    def _check_required_fields(
        fn: str,
        result: Dict[str, Any],
        rules: Dict[str, Any],
        role: Optional[str],
    ) -> None:
        required: List[str] = rules.get("required_fields", [])
        missing = [f for f in required if f not in result]
        if missing:
            raise GuardrailViolation(
                function_name=fn,
                rule="required_fields",
                detail=f"Missing required fields: {missing}",
                role=role,
            )

    @staticmethod
    def _check_redacted_fields(
        fn: str,
        result: Dict[str, Any],
        rules: Dict[str, Any],
        role: Optional[str],
    ) -> None:
        redacted: List[str] = rules.get("redacted_fields", [])
        present = [f for f in redacted if f in result]
        if present:
            raise GuardrailViolation(
                function_name=fn,
                rule="redacted_fields",
                detail=f"Redacted fields must be removed before returning: {present}",
                role=role,
            )
