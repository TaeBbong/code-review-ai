# app/exceptions/handlers.py
from __future__ import annotations

import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from backend.shared.context import run_id_var

logger = logging.getLogger("reviewbot")


def register_exception_handlers(app) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        run_id = run_id_var.get()
        logger.warning(
            "VALIDATION run_id=%s path=%s errors=%s",
            run_id,
            request.url.path,
            exc.errors(),
        )
        return JSONResponse(status_code=422, content={"detail": exc.errors(), "run_id": run_id})
