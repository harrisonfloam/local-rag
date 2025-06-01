import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as api_router
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


app = FastAPI(title="Local RAG API", debug=settings.DEBUG, lifespan=lifespan)

app.include_router(api_router, prefix=settings.API_PREFIX)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.main:app", host=settings.HOST, port=settings.PORT, reload=True)
