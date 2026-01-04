from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.models.requests import ChatCompletionRequest
from app.models.responses import ChatCompletionResponse
from app.core.exceptions import ModelNotFoundError
from app.core.inference import InferenceService
from app.config.settings import settings
import uuid
import time
import json

router = APIRouter()


async def stream_chat_response(
    inference_service: InferenceService,
    request: ChatCompletionRequest,
    completion_id: str,
    created: int
):
    """SSE 스트리밍 응답 생성"""
    # generate_chat_completion is an async function that returns an async generator
    # when `stream=True`. We must await it first to obtain the generator before
    # iterating.
    gen = await inference_service.generate_chat_completion(
        messages=[msg.model_dump() for msg in request.messages],
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        stream=True,
        stop=request.stop,
    )
    async for chunk in gen:
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": chunk["delta"],
                    "finish_reason": chunk.get("finish_reason")
                }
            ]
        }

        yield f"data: {json.dumps(data)}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """채팅 완성 생성 (스트리밍/논스트리밍)"""

    # 모델 검증
    if request.model != settings.MODEL_ID:
        raise ModelNotFoundError(request.model)

    # 추론 서비스 생성
    inference_service = InferenceService()
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    if request.stream:
        # 스트리밍 응답
        return StreamingResponse(
            stream_chat_response(inference_service, request, completion_id, created),
            media_type="text/event-stream"
        )
    else:
        # 논스트리밍 응답
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
