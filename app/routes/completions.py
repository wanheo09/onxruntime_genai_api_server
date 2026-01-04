from fastapi import APIRouter, HTTPException
from app.models.requests import CompletionRequest
from app.models.responses import CompletionResponse
from app.core.exceptions import ModelNotFoundError
from app.config.settings import settings
import uuid
import time

router = APIRouter()


@router.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """텍스트 완성 생성 (논스트리밍 모의 응답)"""

    # 모델 검증
    if request.model != settings.MODEL_ID:
        raise ModelNotFoundError(request.model)

    # 스트리밍은 나중에 구현
    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming not implemented yet")

    # 모의 응답 생성
    completion_id = f"cmpl-{uuid.uuid4()}"
    created = int(time.time())

    mock_response = " there was a kingdom far away..."

    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "text": mock_response,
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15
        }
    }
