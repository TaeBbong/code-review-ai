from fastapi import FastAPI

from backend.api.routes import router
from backend.shared.logging import setup_logging
from backend.middleware.request_context import RequestContextMiddleware
from backend.exceptions.handlers import register_exception_handlers


def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(title="ReviewBot", version="0.1.0")
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()
