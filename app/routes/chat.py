from fastapi import APIRouter, HTTPException
from app.models.requests import ChatCompletionRequest
from app.models.responses import ChatCompletionResponse
from app.core.exceptions import ModelNotFoundError
from app.config.settings import settings
import uuid
import time

router = APIRouter()


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """채팅 완성 생성 (논스트리밍 모의 응답)"""

    # 모델 검증
    if request.model != settings.MODEL_ID:
        raise ModelNotFoundError(request.model)

    # 스트리밍은 나중에 구현
    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming not implemented yet")

    # 모의 응답 생성
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    # 간단한 모의 응답
    mock_response = "Hello! I'm a mock response. Model inference will be implemented in the next phase."

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
                    "content": mock_response
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,  # 모의 값
            "completion_tokens": 15,  # 모의 값
            "total_tokens": 25
        }
    }
