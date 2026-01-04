from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config.settings import settings
from app.core.exceptions import APIException
from app.routes import models, chat, completions
from app.utils.logger import setup_logging

# 로깅 설정
setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ONNX Runtime GenAI OpenAI-Compatible API",
    version="1.0.0",
    description="OpenAI-compatible API server using ONNX Runtime GenAI"
)

# CORS 설정
origins = settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(models.router, tags=["Models"])
app.include_router(chat.router, tags=["Chat"])
app.include_router(completions.router, tags=["Completions"])


# 전역 예외 핸들러
@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    """API 예외 처리"""
    logger.error(f"API Exception: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.type,
                "code": exc.code,
                "param": exc.param
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """일반 예외 처리"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "code": "internal_error",
                "param": None
            }
        }
    )


@app.get("/")
async def root():
    """헬스 체크"""
    return {
        "status": "ok",
        "message": "ONNX Runtime GenAI API Server is running"
    }
