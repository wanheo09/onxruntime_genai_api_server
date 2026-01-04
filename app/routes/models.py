from fastapi import APIRouter
from app.models.responses import ModelListResponse
from app.config.settings import settings
import time

router = APIRouter()

MODEL_CREATED_TIME = int(time.time())


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """사용 가능한 모델 목록 반환"""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.MODEL_ID,
                "object": "model",
                "created": MODEL_CREATED_TIME,
                "owned_by": "system"
            }
        ]
    }
