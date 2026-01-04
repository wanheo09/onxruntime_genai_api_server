from fastapi import APIRouter, HTTPException
from app.models.requests import ChatCompletionRequest
from app.models.responses import ChatCompletionResponse
from app.core.exceptions import ModelNotFoundError
from app.core.inference import InferenceService
from app.config.settings import settings
import uuid
import time

router = APIRouter()


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """채팅 완성 생성"""

    # 모델 검증
    if request.model != settings.MODEL_ID:
        raise ModelNotFoundError(request.model)

    # 스트리밍은 Phase 5에서 구현
    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming not implemented yet")

    # 추론 서비스 생성
    inference_service = InferenceService()

    # 논스트리밍 생성
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    result = await inference_service.generate_chat_completion(
        messages=[msg.model_dump() for msg in request.messages],
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        stream=False,
        stop=request.stop
    )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["content"]
                },
                "finish_reason": result["finish_reason"]
            }
        ],
        "usage": result["usage"]
    }
