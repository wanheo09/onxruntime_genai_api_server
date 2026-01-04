import asyncio
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.exceptions import RateLimitExceededError
import logging

logger = logging.getLogger(__name__)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """동시 요청 수 제한 미들웨어"""

    def __init__(self, app, max_concurrent: int):
        super().__init__(app)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        logger.info(f"Rate limiter initialized with max {max_concurrent} concurrent requests")

    async def dispatch(self, request: Request, call_next):
        # 세마포어 획득 시도
        if self.semaphore.locked():
            logger.warning(f"Rate limit exceeded for {request.url.path}")
            raise RateLimitExceededError()

        async with self.semaphore:
            response = await call_next(request)
            return response
