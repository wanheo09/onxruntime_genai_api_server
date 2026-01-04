from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.models.requests import CompletionRequest
from app.models.responses import CompletionResponse
from app.core.exceptions import ModelNotFoundError
from app.core.inference import InferenceService
from app.config.settings import settings
import uuid
import time
import json

router = APIRouter()


async def stream_completion_response(
    inference_service: InferenceService,
    request: CompletionRequest,
    completion_id: str,
    created: int
):
    """SSE 스트리밍 응답 생성"""
    first_chunk = True

    async for chunk in inference_service.generate_text_completion(
        prompt=request.prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        stream=True
    ):
        # 텍스트 완성은 delta가 아니라 text 필드 사용
        if first_chunk:
            text = chunk["delta"].get("content", "")
            first_chunk = False
        else:
            text = chunk["delta"].get("content", "")

        data = {
            "id": completion_id,
            "object": "text_completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "finish_reason": chunk.get("finish_reason")
                }
            ]
        }

        yield f"data: {json.dumps(data)}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """텍스트 완성 생성 (스트리밍/논스트리밍)"""

    # 모델 검증
    if request.model != settings.MODEL_ID:
        raise ModelNotFoundError(request.model)

    # 추론 서비스 생성
    inference_service = InferenceService()
    completion_id = f"cmpl-{uuid.uuid4()}"
    created = int(time.time())

    if request.stream:
        # 스트리밍 응답
        return StreamingResponse(
            stream_completion_response(inference_service, request, completion_id, created),
            media_type="text/event-stream"
        )
    else:
        # 논스트리밍 응답
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
