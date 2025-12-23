from fastapi import FastAPI
from contextlib import asynccontextmanager

from backend.api.routes import router
from backend.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


app = FastAPI(title="ReviewBot", version="0.1.0", lifespan=lifespan)
app.include_router(router)
