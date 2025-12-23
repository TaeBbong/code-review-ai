from __future__ import annotations
from contextvars import ContextVar

run_id_var: ContextVar[str] = ContextVar("run_id", default="unknown")
