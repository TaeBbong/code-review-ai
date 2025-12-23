from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from backend.core.context import run_id_var

logger = logging.getLogger("reviewbot")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        run_id = str(uuid.uuid4())

        token = run_id_var.set(run_id)

        start = time.time()
        try:
            response = await call_next(request)
        finally:
            run_id_var.reset(token)

        elapsed_ms = (time.time() - start) * 1000

        response.headers["X-Run-Id"] = run_id

        logger.info(
            "REQ run_id=%s %s %s status=%s elapsed=%.1fms",
            run_id,
            request.method,
            request.url.path,
            getattr(response, "status_code", "?"),
            elapsed_ms,
        )
        return response
