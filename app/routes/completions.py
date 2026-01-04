from fastapi import APIRouter, HTTPException
from app.models.requests import CompletionRequest
from app.models.responses import CompletionResponse
from app.core.exceptions import ModelNotFoundError
from app.core.inference import InferenceService
from app.config.settings import settings
import uuid
import time

router = APIRouter()


@router.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """텍스트 완성 생성"""

    # 모델 검증
    if request.model != settings.MODEL_ID:
        raise ModelNotFoundError(request.model)

    # 스트리밍은 Phase 5에서 구현
    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming not implemented yet")

    # 추론 서비스 생성
    inference_service = InferenceService()

    # 논스트리밍 생성
    completion_id = f"cmpl-{uuid.uuid4()}"
    created = int(time.time())

    result = await inference_service.generate_text_completion(
        prompt=request.prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        stream=False
    )

    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "text": result["content"],
                "index": 0,
                "finish_reason": result["finish_reason"]
            }
        ],
        "usage": result["usage"]
    }
