import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.middleware import RequestLoggingMiddleware
from app.api.routes import router
from app.settings import settings
from app.utils.utils import init_logging

init_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    # Startup tasks
    yield
    # Shutdown tasks
    logger.info("Shutting down...")


app = FastAPI(title="Local RAG API", debug=settings.debug, lifespan=lifespan)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(router, prefix=settings.api_prefix)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )
