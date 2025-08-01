import json
import logging
import time
from typing import Callable

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.settings import settings

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP requests and responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        endpoint = request.url.path or "root"
        logger.info(f"{request.method} {endpoint} request received")

        # Log POST request payloads
        if (
            request.method == "POST"
            and request.headers.get("content-type") == "application/json"
        ):
            await self._log_request_body(request, endpoint)

        start_time = time.time()

        try:
            response = await call_next(request)
            duration = time.time() - start_time

            # Handle streaming responses differently
            if hasattr(response, "__class__") and "StreamingResponse" in str(
                response.__class__
            ):
                # This is a streaming response, don't log total time as it's misleading
                logger.info(
                    f"{request.method} {endpoint} {response.status_code} stream started in {duration:.4f}s"
                )
            else:
                logger.info(
                    f"{request.method} {endpoint} {response.status_code} completed in {duration:.4f}s"
                )
            return response

        except HTTPException as e:
            duration = time.time() - start_time
            logger.warning(
                f"{request.method} {endpoint} {e.status_code} HTTP error in {duration:.3f}s: {e.detail}"
            )
            raise

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"{request.method} {endpoint} 500 Internal Error in {duration:.3f}s: {str(e)}",
                exc_info=True,
            )

            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "detail": str(e)
                    if settings.debug
                    else "An unexpected error occurred",
                },
            )

    async def _log_request_body(self, request: Request, endpoint: str) -> None:
        """Log the request body for POST requests."""
        try:
            body = await request.body()
            if body:
                payload = json.loads(body.decode())
                logger.debug(
                    f"{endpoint} request payload:\n{json.dumps(payload, indent=2, default=str)}"
                )
        except Exception as e:
            logger.warning(f"Could not parse request body: {e}")
