import json
import logging
import time

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from app.settings import settings

logger = logging.getLogger(__name__)


async def log_requests(request: Request, call_next):
    """Log HTTP requests and responses."""
    endpoint = request.url.path or "root"
    logger.info(f"{request.method} {endpoint} request received")

    # POST requests
    if (
        request.method == "POST"
        and request.headers.get("content-type") == "application/json"
    ):
        try:
            body = await request.body()
            if body:
                payload = json.loads(body.decode())
                logger.debug(
                    f"{endpoint} request payload:\n{json.dumps(payload, indent=2, default=str)}"
                )
        except Exception as e:
            logger.warning(f"Could not parse request body: {e}")

    start_time = time.time()
    try:
        # Call next middleware or endpoint
        response = await call_next(request)
        duration = time.time() - start_time

        logger.info(
            f"{request.method} {endpoint} {response.status_code} completed in {duration:.4f}s"
        )
        return response

    except HTTPException as e:
        # Expected errors
        duration = time.time() - start_time
        logger.warning(
            f"{request.method} {endpoint} {e.status_code} HTTP error in {duration:.3f}s: {e.detail}"
        )
        raise

    except Exception as e:
        # Unexpected errors
        duration = time.time() - start_time
        logger.error(
            f"{request.method} {endpoint} 500 Internal Error in {duration:.3f}s: {str(e)}",
            exc_info=True,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(e) if settings.debug else "An unexpected error occurred",
            },
        )
