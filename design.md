# ONNX Runtime GenAI API 서버 - 상세 설계 문서

## 1. 개요

본 문서는 [requirements.md](requirements.md)와 [architecture.md](architecture.md)를 기반으로 한 상세 설계 명세입니다.

## 2. 시스템 아키텍처

### 2.1 레이어 구조

```
┌─────────────────────────────────────────┐
│         FastAPI Application             │
│  (uvicorn ASGI Server)                  │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│         Middleware Layer                │
│  - CORS                                 │
│  - Rate Limiter (Semaphore)             │
│  - Request/Response Logging             │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│         API Routes Layer                │
│  - GET /v1/models                       │
│  - POST /v1/chat/completions            │
│  - POST /v1/completions                 │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│      Service Layer (Business Logic)     │
│  - Parameter Validation                 │
│  - Prompt Template Application          │
│  - Inference Orchestration              │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│         Core Layer                      │
│  - ModelLoader (Singleton)              │
│  - Tokenizer Wrapper                    │
│  - ONNX Runtime GenAI Interface         │
└─────────────────────────────────────────┘
```

### 2.2 디렉터리 구조

```
onxruntime_genai_api_server/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI 앱 진입점 및 lifespan 이벤트
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py            # Pydantic Settings (환경 변수)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py            # Request 스키마
│   │   ├── responses.py           # Response 스키마
│   │   └── errors.py              # Error 스키마
│   ├── core/
│   │   ├── __init__.py
│   │   ├── model_loader.py        # 모델 로더 (싱글톤)
│   │   ├── tokenizer.py           # 토크나이저 래퍼
│   │   ├── inference.py           # 추론 엔진
│   │   ├── prompt_builder.py      # Phi-3.5 프롬프트 템플릿
│   │   └── exceptions.py          # 커스텀 예외
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── rate_limiter.py        # 동시 요청 제한
│   │   └── logging_middleware.py  # 요청/응답 로깅
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── models.py              # /v1/models
│   │   ├── chat.py                # /v1/chat/completions
│   │   └── completions.py         # /v1/completions
│   └── utils/
│       ├── __init__.py
│       ├── logger.py              # 로깅 설정
│       └── helpers.py             # 유틸리티 함수
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # pytest fixtures
│   ├── test_models.py
│   ├── test_chat.py
│   ├── test_completions.py
│   └── test_error_handling.py
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
└── run.py                         # 로컬 실행 스크립트
```

## 3. 주요 컴포넌트 설계

### 3.1 설정 관리 (`app/config/settings.py`)

**목적**: 환경 변수를 관리하고 타입 안전성 보장

**구현**:
```python
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # 모델 설정
    MODEL_PATH: str = "/home/wan/Downloads/models/Phi-3.5-mini-instruct-onnx/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4"
    MODEL_ID: str = "phi-3.5-mini"

    # 서버 설정
    SERVER_PORT: int = 8000
    SERVER_HOST: str = "0.0.0.0"

    # 성능 설정
    MAX_CONCURRENT_REQUESTS: int = 10
    DEFAULT_MAX_TOKENS: int = 1024
    DEFAULT_TEMPERATURE: float = 0.7

    # 보안 설정
    CORS_ORIGINS: str = "*"  # 쉼표로 구분된 문자열
    MAX_REQUEST_SIZE_MB: int = 10

    # 로깅 설정
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

### 3.2 모델 로더 (`app/core/model_loader.py`)

**목적**: ONNX Runtime GenAI 모델을 로드하고 전역에서 공유

**책임**:
- 서버 시작 시 모델 로드
- 로드 실패 시 예외 발생 및 서버 종료
- 싱글톤 패턴으로 모델 인스턴스 관리

**주요 메서드**:
```python
class ModelLoader:
    _instance = None
    _model = None
    _tokenizer = None

    @classmethod
    def get_instance(cls):
        """싱글톤 인스턴스 반환"""

    @classmethod
    async def initialize(cls, model_path: str):
        """모델 및 토크나이저 로드"""
        # onnxruntime_genai.Model() 사용
        # 실패 시 ModelLoadingError 발생

    @classmethod
    def get_model(cls):
        """로드된 모델 반환"""

    @classmethod
    def get_tokenizer(cls):
        """로드된 토크나이저 반환"""
```

**오류 처리**:
- 모델 파일이 없거나 손상된 경우 → `ModelLoadingError`
- 서버 시작 시 로그 출력 및 종료

### 3.3 토크나이저 (`app/core/tokenizer.py`)

**목적**: 토큰 수 계산 및 인코딩/디코딩

**주요 메서드**:
```python
class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str) -> List[int]:
        """텍스트를 토큰 ID로 변환"""

    def decode(self, tokens: List[int]) -> str:
        """토큰 ID를 텍스트로 변환"""

    def count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        return len(self.encode(text))
```

### 3.4 프롬프트 빌더 (`app/core/prompt_builder.py`)

**목적**: Phi-3.5 모델의 채팅 템플릿 적용

**Phi-3.5 템플릿 형식**:
```
<|system|>
{system_message}<|end|>
<|user|>
{user_message}<|end|>
<|assistant|>
{assistant_message}<|end|>
```

**주요 메서드**:
```python
class Phi35PromptBuilder:
    @staticmethod
    def build_chat_prompt(messages: List[dict]) -> str:
        """메시지 배열을 Phi-3.5 프롬프트로 변환"""
        # messages 검증
        # 각 role에 따라 템플릿 적용
        # 마지막에 <|assistant|> 추가

    @staticmethod
    def build_completion_prompt(prompt: str) -> str:
        """단순 텍스트 완성 프롬프트"""
        # 템플릿 없이 그대로 반환
```

### 3.5 추론 엔진 (`app/core/inference.py`)

**목적**: ONNX Runtime GenAI를 사용한 텍스트 생성

**주요 클래스**:
```python
class InferenceService:
    def __init__(self, model_loader: ModelLoader, tokenizer: TokenizerWrapper):
        self.model = model_loader.get_model()
        self.tokenizer = tokenizer

    async def generate_chat_completion(
        self,
        messages: List[dict],
        temperature: float,
        max_tokens: int,
        top_p: float,
        stream: bool,
        stop: Optional[List[str]] = None,
        **kwargs
    ):
        """채팅 완성 생성 (스트리밍/논스트리밍)"""

    async def generate_text_completion(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stream: bool,
        **kwargs
    ):
        """텍스트 완성 생성"""

    async def _generate_streaming(self, ...):
        """스트리밍 생성 (async generator)"""
        # yield 로 청크 반환

    async def _generate_non_streaming(self, ...):
        """논스트리밍 생성"""
        # 전체 텍스트 한 번에 반환
```

**ONNX Runtime GenAI 사용 예**:
```python
import onnxruntime_genai as og

# 모델 로드
model = og.Model(model_path)
tokenizer = og.Tokenizer(model)

# 생성 파라미터 설정
params = og.GeneratorParams(model)
params.set_search_options(
    max_length=max_tokens,
    temperature=temperature,
    top_p=top_p
)
params.input_ids = tokenizer.encode(prompt)

# 생성
generator = og.Generator(model, params)
while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token()

    # 스트리밍: 새 토큰 반환
    new_token = generator.get_next_tokens()[0]
    yield tokenizer.decode([new_token])
```

### 3.6 Rate Limiter 미들웨어 (`app/middleware/rate_limiter.py`)

**목적**: 동시 요청 수 제한

**구현**:
```python
import asyncio
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_concurrent: int):
        super().__init__(app)
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def dispatch(self, request: Request, call_next):
        if not self.semaphore.locked():
            async with self.semaphore:
                return await call_next(request)
        else:
            # 429 응답
            raise RateLimitExceededError()
```

### 3.7 로깅 미들웨어 (`app/middleware/logging_middleware.py`)

**목적**: 요청/응답 로깅 및 추론 시간 측정

**로깅 내용**:
- 요청 시작: method, path, params
- 응답 완료: status_code, response_time_ms
- 추론 시간: inference_time_ms
- 에러: error message, stack trace

**구현**:
```python
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # 요청 로깅
        logger.info(f"Request: {request.method} {request.url.path}")

        response = await call_next(request)

        # 응답 로깅
        duration = (time.time() - start_time) * 1000
        logger.info(f"Response: {response.status_code} ({duration:.2f}ms)")

        return response
```

### 3.8 Request/Response 스키마 (`app/models/`)

**requests.py**:
```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union

class Message(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1024, gt=0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    frequency_penalty: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None

    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        return v

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=100, gt=0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = False
```

**responses.py**:
```python
from pydantic import BaseModel
from typing import List, Optional

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str  # "stop", "length", "content_filter"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage

class StreamChatChoice(BaseModel):
    index: int
    delta: dict  # {"role": "assistant", "content": "..."} or {"content": "..."}
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChatChoice]
```

**errors.py**:
```python
from pydantic import BaseModel
from typing import Optional

class ErrorDetail(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: str

class ErrorResponse(BaseModel):
    error: ErrorDetail
```

### 3.9 커스텀 예외 (`app/core/exceptions.py`)

```python
class APIException(Exception):
    def __init__(self, message: str, type: str, code: str, param: str = None, status_code: int = 400):
        self.message = message
        self.type = type
        self.code = code
        self.param = param
        self.status_code = status_code

class ModelNotFoundError(APIException):
    def __init__(self, model: str):
        super().__init__(
            message=f"Model '{model}' not found",
            type="invalid_request_error",
            code="model_not_found",
            param="model",
            status_code=404
        )

class InvalidParameterError(APIException):
    def __init__(self, param: str, message: str):
        super().__init__(
            message=message,
            type="invalid_request_error",
            code="invalid_parameter",
            param=param,
            status_code=400
        )

class RateLimitExceededError(APIException):
    def __init__(self):
        super().__init__(
            message="Too many concurrent requests. Please try again later.",
            type="rate_limit_error",
            code="rate_limit_exceeded",
            status_code=429
        )

class ModelLoadingError(APIException):
    def __init__(self, path: str, detail: str):
        super().__init__(
            message=f"Failed to load model from {path}: {detail}",
            type="server_error",
            code="model_loading_failed",
            status_code=500
        )
```

## 4. API 엔드포인트 구현

### 4.1 GET /v1/models (`app/routes/models.py`)

```python
from fastapi import APIRouter
from app.models.responses import ModelListResponse
from app.config.settings import settings
import time

router = APIRouter()

MODEL_CREATED_TIME = int(time.time())

@router.get("/v1/models")
async def list_models():
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
```

### 4.2 POST /v1/chat/completions (`app/routes/chat.py`)

```python
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.requests import ChatCompletionRequest
from app.core.inference import InferenceService
from app.core.exceptions import ModelNotFoundError
from app.config.settings import settings
import uuid
import time
import json

router = APIRouter()

@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # 1. 모델 검증
    if request.model != settings.MODEL_ID:
        raise ModelNotFoundError(request.model)

    # 2. 추론 서비스 호출
    inference_service = InferenceService.get_instance()

    # 3. 스트리밍 vs 논스트리밍
    if request.stream:
        return StreamingResponse(
            stream_chat_response(inference_service, request),
            media_type="text/event-stream"
        )
    else:
        return await generate_chat_response(inference_service, request)

async def stream_chat_response(service, request):
    """SSE 스트리밍 응답 생성"""
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    async for chunk in service.generate_chat_completion(
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        stream=True
    ):
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": chunk["delta"],
                "finish_reason": chunk.get("finish_reason")
            }]
        }
        yield f"data: {json.dumps(data)}\n\n"

    yield "data: [DONE]\n\n"

async def generate_chat_response(service, request):
    """논스트리밍 응답 생성"""
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    result = await service.generate_chat_completion(
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        stream=False
    )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result["content"]
            },
            "finish_reason": result["finish_reason"]
        }],
        "usage": result["usage"]
    }
```

### 4.3 POST /v1/completions (`app/routes/completions.py`)

구조는 chat.py와 유사하되, `object` 필드와 응답 형식이 다름

## 5. 애플리케이션 진입점 (`app/main.py`)

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config.settings import settings
from app.core.model_loader import ModelLoader
from app.core.exceptions import APIException, ModelLoadingError
from app.middleware.rate_limiter import RateLimiterMiddleware
from app.middleware.logging_middleware import LoggingMiddleware
from app.routes import models, chat, completions
from app.utils.logger import setup_logging

# 로깅 설정
setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 이벤트"""
    # 시작
    logger.info("Starting ONNX Runtime GenAI API Server...")
    try:
        await ModelLoader.initialize(settings.MODEL_PATH)
        logger.info(f"Model loaded successfully from {settings.MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise ModelLoadingError(settings.MODEL_PATH, str(e))

    yield

    # 종료
    logger.info("Shutting down server...")

app = FastAPI(
    title="ONNX Runtime GenAI OpenAI-Compatible API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiter
app.add_middleware(
    RateLimiterMiddleware,
    max_concurrent=settings.MAX_CONCURRENT_REQUESTS
)

# Logging
app.add_middleware(LoggingMiddleware)

# 라우터 등록
app.include_router(models.router)
app.include_router(chat.router)
app.include_router(completions.router)

# 전역 예외 핸들러
@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
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
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "code": "internal_error"
            }
        }
    )
```

## 6. 배포

### 6.1 requirements.txt

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
pydantic==2.9.0
pydantic-settings==2.5.0
onnxruntime==1.20.0
onnxruntime-genai==0.5.0
python-multipart==0.0.9
```

### 6.2 Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드
COPY app/ ./app/

# 환경 변수
ENV PYTHONUNBUFFERED=1

# 포트 노출
EXPOSE 8000

# 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.3 docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/phi-3.5-mini
      - SERVER_PORT=8000
      - MAX_CONCURRENT_REQUESTS=10
      - LOG_LEVEL=INFO
    volumes:
      - /home/wan/Downloads/models/Phi-3.5-mini-instruct-onnx:/models
    restart: unless-stopped
```

## 7. 테스트 전략

### 7.1 단위 테스트
- `test_tokenizer.py`: 토큰 수 계산 검증
- `test_prompt_builder.py`: 프롬프트 템플릿 검증
- `test_validation.py`: 파라미터 검증 로직

### 7.2 통합 테스트
- `test_chat.py`: /v1/chat/completions 엔드포인트
- `test_completions.py`: /v1/completions 엔드포인트
- `test_streaming.py`: SSE 스트리밍 검증

### 7.3 에러 핸들링 테스트
- `test_error_handling.py`: 모든 에러 케이스 검증

## 8. 확장 가능성

### 8.1 인증 추가
- `app/middleware/auth.py` 생성
- API 키 검증 로직 추가

### 8.2 멀티모달 지원
- `app/routes/images.py` 추가
- 이미지 인코딩 로직 구현

### 8.3 임베딩 API
- 별도 임베딩 모델 로드
- `app/routes/embeddings.py` 구현

## 9. 구현 순서 제안

1. **기본 구조 설정** (1-2일)
   - 디렉터리 구조 생성
   - settings.py, main.py 작성
   - 기본 라우터 스켈레톤

2. **모델 로더 및 추론 엔진** (2-3일)
   - ModelLoader 구현
   - Tokenizer 래퍼
   - InferenceService 기본 구현

3. **API 엔드포인트** (3-4일)
   - GET /v1/models
   - POST /v1/chat/completions (논스트리밍)
   - POST /v1/completions (논스트리밍)

4. **스트리밍 지원** (2-3일)
   - SSE 스트리밍 구현
   - 청크 생성 로직

5. **에러 처리 및 검증** (1-2일)
   - 모든 에러 케이스 구현
   - 파라미터 검증 강화

6. **미들웨어 및 비기능 요구사항** (2-3일)
   - Rate Limiter
   - Logging
   - CORS

7. **테스트 작성** (2-3일)
   - 단위 테스트
   - 통합 테스트

8. **배포 준비** (1일)
   - Dockerfile
   - docker-compose.yml
   - README.md

**총 예상 기간**: 약 2-3주
