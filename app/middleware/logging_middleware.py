import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """요청/응답 로깅 미들웨어"""

    async def dispatch(self, request: Request, call_next):
        # 요청 시작
        start_time = time.time()

        # 요청 정보 로깅
        logger.info(
            f"Request started: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )

        # 요청 처리
        response = await call_next(request)

        # 응답 시간 계산
        duration_ms = (time.time() - start_time) * 1000

        # 응답 정보 로깅
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"status={response.status_code} duration={duration_ms:.2f}ms"
        )

        # 응답 헤더에 처리 시간 추가
        response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"

        return response
