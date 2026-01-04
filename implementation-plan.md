# ONNX Runtime GenAI API 서버 구현 계획

## 목차
1. [Phase 1: 프로젝트 초기 설정](#phase-1-프로젝트-초기-설정)
2. [Phase 2: 핵심 인프라 구현](#phase-2-핵심-인프라-구현)
3. [Phase 3: API 엔드포인트 구현 (논스트리밍)](#phase-3-api-엔드포인트-구현-논스트리밍)
4. [Phase 4: 스트리밍 기능 구현](#phase-4-스트리밍-기능-구현)
5. [Phase 5: 에러 처리 및 검증 강화](#phase-5-에러-처리-및-검증-강화)
6. [Phase 6: 미들웨어 및 비기능 요구사항](#phase-6-미들웨어-및-비기능-요구사항)
7. [Phase 7: 테스트 작성](#phase-7-테스트-작성)
8. [Phase 8: 배포 준비](#phase-8-배포-준비)

---

## Phase 1: 프로젝트 초기 설정

### 목표
프로젝트 구조를 생성하고 기본 개발 환경을 설정합니다.

### 작업 목록

#### 1.1 디렉터리 구조 생성
```bash
mkdir -p app/{config,models,core,middleware,routes,utils}
mkdir -p tests
touch app/__init__.py
touch app/config/__init__.py
touch app/models/__init__.py
touch app/core/__init__.py
touch app/middleware/__init__.py
touch app/routes/__init__.py
touch app/utils/__init__.py
touch tests/__init__.py
```

**생성할 파일**:
- `app/__init__.py`
- `app/config/__init__.py`, `app/config/settings.py`
- `app/models/__init__.py`, `app/models/requests.py`, `app/models/responses.py`, `app/models/errors.py`
- `app/core/__init__.py`, `app/core/exceptions.py`
- `app/middleware/__init__.py`
- `app/routes/__init__.py`
- `app/utils/__init__.py`, `app/utils/logger.py`
- `app/main.py`
- `tests/__init__.py`, `tests/conftest.py`

#### 1.2 의존성 설치 파일 작성
**파일**: `requirements.txt`
```
fastapi==0.115.0
uvicorn[standard]==0.30.0
pydantic==2.9.0
pydantic-settings==2.5.0
onnxruntime==1.20.0
onnxruntime-genai==0.5.0
python-multipart==0.0.9
pytest==8.3.0
pytest-asyncio==0.24.0
httpx==0.27.0
```

#### 1.3 환경 변수 예시 파일 작성
**파일**: `.env.example`
```env
# 모델 설정
MODEL_PATH=/home/wan/Downloads/models/Phi-3.5-mini-instruct-onnx/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4
MODEL_ID=phi-3.5-mini

# 서버 설정
SERVER_PORT=8000
SERVER_HOST=0.0.0.0

# 성능 설정
MAX_CONCURRENT_REQUESTS=10
DEFAULT_MAX_TOKENS=1024
DEFAULT_TEMPERATURE=0.7

# 보안 설정
CORS_ORIGINS=*
MAX_REQUEST_SIZE_MB=10

# 로깅 설정
LOG_LEVEL=INFO
```

#### 1.4 기본 실행 스크립트 작성
**파일**: `run.py`
```python
import uvicorn
from app.config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
```

### 완료 조건
- [ ] 모든 디렉터리 및 `__init__.py` 생성
- [ ] `requirements.txt` 작성 완료
- [ ] `.env.example` 작성 완료
- [ ] `run.py` 작성 완료
- [ ] 의존성 설치 확인: `pip install -r requirements.txt`

---

## Phase 2: 핵심 인프라 구현

### 목표
모델 로딩, 설정 관리, 로깅 등 핵심 인프라를 구축합니다.

### 작업 목록

#### 2.1 설정 관리 구현
**파일**: `app/config/settings.py`

```python
from pydantic_settings import BaseSettings
from typing import Optional

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
    CORS_ORIGINS: str = "*"
    MAX_REQUEST_SIZE_MB: int = 10

    # 로깅 설정
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

**테스트**: 환경 변수 로딩 확인
```python
from app.config.settings import settings
print(f"Model Path: {settings.MODEL_PATH}")
print(f"Server Port: {settings.SERVER_PORT}")
```

#### 2.2 로깅 유틸리티 구현
**파일**: `app/utils/logger.py`

```python
import logging
import sys
from typing import Optional

def setup_logging(log_level: str = "INFO"):
    """로깅 설정"""
    level = getattr(logging, log_level.upper(), logging.INFO)

    # 포매터 설정
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    return root_logger

def get_logger(name: str) -> logging.Logger:
    """이름으로 로거 가져오기"""
    return logging.getLogger(name)
```

#### 2.3 커스텀 예외 구현
**파일**: `app/core/exceptions.py`

```python
from typing import Optional

class APIException(Exception):
    """기본 API 예외 클래스"""
    def __init__(
        self,
        message: str,
        type: str,
        code: str,
        param: Optional[str] = None,
        status_code: int = 400
    ):
        self.message = message
        self.type = type
        self.code = code
        self.param = param
        self.status_code = status_code
        super().__init__(self.message)

class ModelNotFoundError(APIException):
    """모델을 찾을 수 없음"""
    def __init__(self, model: str):
        super().__init__(
            message=f"Model '{model}' not found",
            type="invalid_request_error",
            code="model_not_found",
            param="model",
            status_code=404
        )

class InvalidParameterError(APIException):
    """잘못된 파라미터"""
    def __init__(self, param: str, message: str):
        super().__init__(
            message=message,
            type="invalid_request_error",
            code="invalid_parameter",
            param=param,
            status_code=400
        )

class InvalidMessagesError(APIException):
    """잘못된 메시지 형식"""
    def __init__(self, message: str = "Invalid messages format"):
        super().__init__(
            message=message,
            type="invalid_request_error",
            code="invalid_messages",
            param="messages",
            status_code=400
        )

class ContextLengthExceededError(APIException):
    """컨텍스트 길이 초과"""
    def __init__(self, param: str = "messages"):
        super().__init__(
            message="Context length exceeded",
            type="invalid_request_error",
            code="context_length_exceeded",
            param=param,
            status_code=400
        )

class MissingParameterError(APIException):
    """필수 파라미터 누락"""
    def __init__(self, param: str):
        super().__init__(
            message=f"Missing required parameter: {param}",
            type="invalid_request_error",
            code="missing_parameter",
            param=param,
            status_code=400
        )

class RateLimitExceededError(APIException):
    """요청 제한 초과"""
    def __init__(self):
        super().__init__(
            message="Too many concurrent requests. Please try again later.",
            type="rate_limit_error",
            code="rate_limit_exceeded",
            status_code=429
        )

class ModelLoadingError(APIException):
    """모델 로딩 실패"""
    def __init__(self, path: str, detail: str):
        super().__init__(
            message=f"Failed to load model from {path}: {detail}",
            type="server_error",
            code="model_loading_failed",
            status_code=500
        )

class InternalServerError(APIException):
    """서버 내부 오류"""
    def __init__(self, message: str = "Internal server error"):
        super().__init__(
            message=message,
            type="server_error",
            code="internal_error",
            status_code=500
        )
```

#### 2.4 스키마 모델 구현
**파일**: `app/models/errors.py`

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

**파일**: `app/models/requests.py`

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union

class Message(BaseModel):
    role: str
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

    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        for msg in v:
            if msg.role not in ['system', 'user', 'assistant']:
                raise ValueError(f"Invalid role: {msg.role}")
        return v

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=100, gt=0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    frequency_penalty: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
```

**파일**: `app/models/responses.py`

```python
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

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
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage

class StreamDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class StreamChatChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChatChoice]

class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: str

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "system"

class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]
```

### 완료 조건
- [ ] `settings.py` 구현 및 환경 변수 로딩 테스트
- [ ] `logger.py` 구현 및 로깅 테스트
- [ ] `exceptions.py` 모든 예외 클래스 구현
- [ ] `requests.py`, `responses.py`, `errors.py` 스키마 구현
- [ ] Pydantic 검증 테스트

---

## Phase 3: API 엔드포인트 구현 (논스트리밍)

### 목표
논스트리밍 모드의 API 엔드포인트를 구현합니다. 이 단계에서는 실제 모델 추론 대신 모의 응답을 반환합니다.

### 작업 목록

#### 3.1 모델 목록 API 구현
**파일**: `app/routes/models.py`

```python
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
```

#### 3.2 채팅 완성 API 구현 (모의 응답)
**파일**: `app/routes/chat.py`

```python
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
```

#### 3.3 텍스트 완성 API 구현 (모의 응답)
**파일**: `app/routes/completions.py`

```python
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
```

#### 3.4 FastAPI 애플리케이션 통합
**파일**: `app/main.py`

```python
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
```

#### 3.5 테스트
```bash
# 서버 실행
python run.py

# 테스트 (다른 터미널에서)
curl http://localhost:8000/
curl http://localhost:8000/v1/models
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-3.5-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 완료 조건
- [ ] GET /v1/models 구현 및 테스트
- [ ] POST /v1/chat/completions (모의 응답) 구현 및 테스트
- [ ] POST /v1/completions (모의 응답) 구현 및 테스트
- [ ] FastAPI 앱 통합 및 예외 핸들러 테스트
- [ ] 잘못된 모델 ID 요청 시 404 오류 반환 확인

---

## Phase 4: 모델 로더 및 추론 엔진 구현

### 목표
ONNX Runtime GenAI를 사용하여 실제 모델 로딩 및 추론을 구현합니다.

### 작업 목록

#### 4.1 프롬프트 빌더 구현
**파일**: `app/core/prompt_builder.py`

```python
from typing import List, Dict
from app.core.exceptions import InvalidMessagesError

class Phi35PromptBuilder:
    """Phi-3.5 모델용 프롬프트 템플릿 빌더"""

    SYSTEM_TOKEN = "<|system|>"
    USER_TOKEN = "<|user|>"
    ASSISTANT_TOKEN = "<|assistant|>"
    END_TOKEN = "<|end|>"

    @classmethod
    def build_chat_prompt(cls, messages: List[Dict[str, str]]) -> str:
        """
        메시지 배열을 Phi-3.5 채팅 프롬프트로 변환

        Args:
            messages: [{"role": "user", "content": "..."}, ...]

        Returns:
            Phi-3.5 형식의 프롬프트 문자열
        """
        if not messages:
            raise InvalidMessagesError("messages cannot be empty")

        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"{cls.SYSTEM_TOKEN}\n{content}{cls.END_TOKEN}")
            elif role == "user":
                prompt_parts.append(f"{cls.USER_TOKEN}\n{content}{cls.END_TOKEN}")
            elif role == "assistant":
                prompt_parts.append(f"{cls.ASSISTANT_TOKEN}\n{content}{cls.END_TOKEN}")
            else:
                raise InvalidMessagesError(f"Invalid role: {role}")

        # 마지막에 assistant 토큰 추가
        prompt_parts.append(cls.ASSISTANT_TOKEN)

        return "\n".join(prompt_parts)

    @classmethod
    def build_completion_prompt(cls, prompt: str) -> str:
        """
        단순 텍스트 완성 프롬프트 (템플릿 없음)

        Args:
            prompt: 입력 텍스트

        Returns:
            원본 프롬프트 (템플릿 적용 안 함)
        """
        return prompt
```

#### 4.2 토크나이저 래퍼 구현
**파일**: `app/core/tokenizer.py`

```python
from typing import List
import logging

logger = logging.getLogger(__name__)

class TokenizerWrapper:
    """ONNX Runtime GenAI 토크나이저 래퍼"""

    def __init__(self, tokenizer):
        """
        Args:
            tokenizer: onnxruntime_genai.Tokenizer 인스턴스
        """
        self.tokenizer = tokenizer

    def encode(self, text: str) -> List[int]:
        """
        텍스트를 토큰 ID로 인코딩

        Args:
            text: 입력 텍스트

        Returns:
            토큰 ID 리스트
        """
        try:
            tokens = self.tokenizer.encode(text)
            return tokens
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            raise

    def decode(self, tokens: List[int]) -> str:
        """
        토큰 ID를 텍스트로 디코딩

        Args:
            tokens: 토큰 ID 리스트

        Returns:
            디코딩된 텍스트
        """
        try:
            text = self.tokenizer.decode(tokens)
            return text
        except Exception as e:
            logger.error(f"Detokenization error: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        텍스트의 토큰 수 계산

        Args:
            text: 입력 텍스트

        Returns:
            토큰 수
        """
        tokens = self.encode(text)
        return len(tokens)
```

#### 4.3 모델 로더 구현
**파일**: `app/core/model_loader.py`

```python
import logging
from typing import Optional
try:
    import onnxruntime_genai as og
except ImportError:
    og = None

from app.core.exceptions import ModelLoadingError
from app.core.tokenizer import TokenizerWrapper

logger = logging.getLogger(__name__)

class ModelLoader:
    """ONNX Runtime GenAI 모델 로더 (싱글톤)"""

    _instance: Optional['ModelLoader'] = None
    _model = None
    _tokenizer = None
    _tokenizer_wrapper = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def initialize(cls, model_path: str):
        """
        모델 및 토크나이저 로드

        Args:
            model_path: 모델 디렉터리 경로

        Raises:
            ModelLoadingError: 모델 로딩 실패 시
        """
        if og is None:
            raise ModelLoadingError(
                model_path,
                "onnxruntime_genai not installed. Install with: pip install onnxruntime-genai"
            )

        instance = cls()

        if instance._model is not None:
            logger.info("Model already loaded")
            return instance

        try:
            logger.info(f"Loading model from {model_path}")

            # 모델 로드
            instance._model = og.Model(model_path)
            logger.info("Model loaded successfully")

            # 토크나이저 로드
            instance._tokenizer = og.Tokenizer(instance._model)
            instance._tokenizer_wrapper = TokenizerWrapper(instance._tokenizer)
            logger.info("Tokenizer loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise ModelLoadingError(model_path, str(e))

        return instance

    @classmethod
    def get_instance(cls) -> 'ModelLoader':
        """싱글톤 인스턴스 반환"""
        if cls._instance is None or cls._instance._model is None:
            raise RuntimeError("ModelLoader not initialized. Call initialize() first.")
        return cls._instance

    @classmethod
    def get_model(cls):
        """로드된 모델 반환"""
        instance = cls.get_instance()
        return instance._model

    @classmethod
    def get_tokenizer(cls) -> TokenizerWrapper:
        """토크나이저 래퍼 반환"""
        instance = cls.get_instance()
        return instance._tokenizer_wrapper
```

#### 4.4 추론 엔진 구현
**파일**: `app/core/inference.py`

```python
import logging
from typing import List, Dict, Optional, AsyncGenerator
try:
    import onnxruntime_genai as og
except ImportError:
    og = None

from app.core.model_loader import ModelLoader
from app.core.tokenizer import TokenizerWrapper
from app.core.prompt_builder import Phi35PromptBuilder
from app.core.exceptions import InternalServerError, ContextLengthExceededError

logger = logging.getLogger(__name__)

class InferenceService:
    """추론 서비스"""

    def __init__(self):
        self.model_loader = ModelLoader.get_instance()
        self.model = self.model_loader.get_model()
        self.tokenizer = self.model_loader.get_tokenizer()
        self.prompt_builder = Phi35PromptBuilder()

    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        **kwargs
    ):
        """
        채팅 완성 생성

        Args:
            messages: 메시지 배열
            temperature: 온도 (0.0 ~ 2.0)
            max_tokens: 최대 생성 토큰 수
            top_p: Top-p 샘플링
            stream: 스트리밍 여부
            stop: 정지 시퀀스

        Returns:
            스트리밍: AsyncGenerator
            논스트리밍: dict
        """
        # 프롬프트 빌드
        prompt = self.prompt_builder.build_chat_prompt(messages)
        logger.info(f"Generated prompt (length: {len(prompt)})")

        # 토큰 수 계산
        prompt_tokens = self.tokenizer.count_tokens(prompt)
        logger.info(f"Prompt tokens: {prompt_tokens}")

        if stream:
            return self._generate_streaming(
                prompt, prompt_tokens, temperature, max_tokens, top_p, stop
            )
        else:
            return await self._generate_non_streaming(
                prompt, prompt_tokens, temperature, max_tokens, top_p, stop
            )

    async def generate_text_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 100,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs
    ):
        """
        텍스트 완성 생성

        Args:
            prompt: 입력 프롬프트
            temperature: 온도
            max_tokens: 최대 생성 토큰 수
            top_p: Top-p 샘플링
            stream: 스트리밍 여부

        Returns:
            스트리밍: AsyncGenerator
            논스트리밍: dict
        """
        # 토큰 수 계산
        prompt_tokens = self.tokenizer.count_tokens(prompt)
        logger.info(f"Prompt tokens: {prompt_tokens}")

        if stream:
            return self._generate_streaming(
                prompt, prompt_tokens, temperature, max_tokens, top_p, None
            )
        else:
            return await self._generate_non_streaming(
                prompt, prompt_tokens, temperature, max_tokens, top_p, None
            )

    async def _generate_non_streaming(
        self,
        prompt: str,
        prompt_tokens: int,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: Optional[List[str]]
    ) -> Dict:
        """논스트리밍 생성"""
        try:
            # 생성 파라미터 설정
            params = og.GeneratorParams(self.model)
            params.set_search_options(
                max_length=max_tokens,
                temperature=temperature,
                top_p=top_p
            )

            # 입력 토큰화
            input_tokens = self.tokenizer.encode(prompt)
            params.input_ids = input_tokens

            # 생성
            logger.info("Starting generation...")
            generator = og.Generator(self.model, params)

            generated_tokens = []
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()

                new_token = generator.get_next_tokens()[0]
                generated_tokens.append(new_token)

            # 디코딩
            generated_text = self.tokenizer.decode(generated_tokens)
            completion_tokens = len(generated_tokens)

            logger.info(f"Generation completed. Tokens: {completion_tokens}")

            return {
                "content": generated_text,
                "finish_reason": "stop",  # TODO: length 체크
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }

        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            raise InternalServerError(f"Generation failed: {str(e)}")

    async def _generate_streaming(
        self,
        prompt: str,
        prompt_tokens: int,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: Optional[List[str]]
    ) -> AsyncGenerator:
        """스트리밍 생성"""
        try:
            # 생성 파라미터 설정
            params = og.GeneratorParams(self.model)
            params.set_search_options(
                max_length=max_tokens,
                temperature=temperature,
                top_p=top_p
            )

            # 입력 토큰화
            input_tokens = self.tokenizer.encode(prompt)
            params.input_ids = input_tokens

            # 생성
            logger.info("Starting streaming generation...")
            generator = og.Generator(self.model, params)

            completion_tokens = 0
            first_chunk = True

            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()

                new_token = generator.get_next_tokens()[0]
                completion_tokens += 1

                # 토큰 디코딩
                token_text = self.tokenizer.decode([new_token])

                # 첫 번째 청크는 role 포함
                if first_chunk:
                    yield {
                        "delta": {"role": "assistant", "content": token_text},
                        "finish_reason": None
                    }
                    first_chunk = False
                else:
                    yield {
                        "delta": {"content": token_text},
                        "finish_reason": None
                    }

            # 마지막 청크 (finish_reason 포함)
            yield {
                "delta": {},
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }

            logger.info(f"Streaming completed. Tokens: {completion_tokens}")

        except Exception as e:
            logger.error(f"Streaming generation error: {e}", exc_info=True)
            raise InternalServerError(f"Streaming generation failed: {str(e)}")
```

#### 4.5 FastAPI 앱에 모델 로더 통합
**파일**: `app/main.py` 수정

```python
from contextlib import asynccontextmanager

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
        raise

    yield

    # 종료
    logger.info("Shutting down server...")

# FastAPI 앱 생성 시 lifespan 추가
app = FastAPI(
    title="ONNX Runtime GenAI OpenAI-Compatible API",
    version="1.0.0",
    description="OpenAI-compatible API server using ONNX Runtime GenAI",
    lifespan=lifespan
)
```

#### 4.6 라우터에서 실제 추론 사용
**파일**: `app/routes/chat.py` 수정

```python
from app.core.inference import InferenceService

@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """채팅 완성 생성"""

    # 모델 검증
    if request.model != settings.MODEL_ID:
        raise ModelNotFoundError(request.model)

    # 추론 서비스 생성
    inference_service = InferenceService()

    # 스트리밍 (Phase 5에서 구현)
    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming not implemented yet")

    # 논스트리밍 생성
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    result = await inference_service.generate_chat_completion(
        messages=[msg.dict() for msg in request.messages],
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
```

### 완료 조건
- [ ] 프롬프트 빌더 구현 및 테스트
- [ ] 토크나이저 래퍼 구현 및 테스트
- [ ] 모델 로더 구현 및 초기화 테스트
- [ ] 추론 엔진 (논스트리밍) 구현
- [ ] 실제 모델로 채팅 완성 테스트
- [ ] 토큰 사용량 정확성 확인

---

## Phase 5: 스트리밍 기능 구현

### 목표
SSE (Server-Sent Events)를 사용한 스트리밍 응답을 구현합니다.

### 작업 목록

#### 5.1 채팅 완성 스트리밍 구현
**파일**: `app/routes/chat.py` 수정

```python
from fastapi.responses import StreamingResponse
import json

async def stream_chat_response(
    inference_service: InferenceService,
    request: ChatCompletionRequest,
    completion_id: str,
    created: int
):
    """SSE 스트리밍 응답 생성"""

    async for chunk in inference_service.generate_chat_completion(
        messages=[msg.dict() for msg in request.messages],
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        stream=True,
        stop=request.stop
    ):
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

    if request.model != settings.MODEL_ID:
        raise ModelNotFoundError(request.model)

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
        # 논스트리밍 응답 (기존 코드)
        # ...
```

#### 5.2 텍스트 완성 스트리밍 구현
**파일**: `app/routes/completions.py` 수정

유사한 방식으로 스트리밍 구현

### 완료 조건
- [ ] 채팅 완성 스트리밍 구현
- [ ] 텍스트 완성 스트리밍 구현
- [ ] SSE 형식 검증
- [ ] `data: [DONE]` 메시지 확인
- [ ] 스트리밍 중 finish_reason null 확인
- [ ] 마지막 청크에서 finish_reason 확인

---

## Phase 6: 미들웨어 및 비기능 요구사항

### 목표
Rate Limiter, 로깅 미들웨어 등 비기능 요구사항을 구현합니다.

### 작업 목록

#### 6.1 Rate Limiter 미들웨어 구현
**파일**: `app/middleware/rate_limiter.py`

```python
import asyncio
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.exceptions import RateLimitExceededError
import logging

logger = logging.getLogger(__name__)

class RateLimiterMiddleware(BaseHTTPMiddleware):
    """동시 요청 수 제한 미들웨어"""

    def __init__(self, app, max_concurrent: int):
        super().__init__(app)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        logger.info(f"Rate limiter initialized with max {max_concurrent} concurrent requests")

    async def dispatch(self, request: Request, call_next):
        # 세마포어 획득 시도
        if self.semaphore.locked():
            logger.warning(f"Rate limit exceeded for {request.url.path}")
            raise RateLimitExceededError()

        async with self.semaphore:
            response = await call_next(request)
            return response
```

#### 6.2 로깅 미들웨어 구현
**파일**: `app/middleware/logging_middleware.py`

```python
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
```

#### 6.3 미들웨어 등록
**파일**: `app/main.py` 수정

```python
from app.middleware.rate_limiter import RateLimiterMiddleware
from app.middleware.logging_middleware import LoggingMiddleware

# ... (기존 코드)

# 미들웨어 등록 (순서 중요: 먼저 등록된 것이 나중에 실행)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimiterMiddleware, max_concurrent=settings.MAX_CONCURRENT_REQUESTS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 완료 조건
- [ ] Rate Limiter 구현 및 429 응답 테스트
- [ ] 로깅 미들웨어 구현 및 로그 출력 확인
- [ ] 동시 요청 테스트
- [ ] CORS 설정 확인

---

## Phase 7: 테스트 작성

### 목표
단위 테스트 및 통합 테스트를 작성합니다.

### 작업 목록

#### 7.1 pytest 설정
**파일**: `tests/conftest.py`

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    """테스트 클라이언트"""
    return TestClient(app)

@pytest.fixture
def mock_messages():
    """모의 메시지"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
```

#### 7.2 모델 목록 테스트
**파일**: `tests/test_models.py`

```python
def test_list_models(client):
    """모델 목록 조회 테스트"""
    response = client.get("/v1/models")
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "phi-3.5-mini"
```

#### 7.3 채팅 완성 테스트
**파일**: `tests/test_chat.py`

```python
def test_chat_completion_success(client, mock_messages):
    """채팅 완성 성공 테스트"""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "phi-3.5-mini",
            "messages": mock_messages,
            "temperature": 0.7,
            "max_tokens": 100
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "phi-3.5-mini"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "usage" in data

def test_chat_completion_invalid_model(client, mock_messages):
    """잘못된 모델 ID 테스트"""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "invalid-model",
            "messages": mock_messages
        }
    )

    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "model_not_found"
```

#### 7.4 에러 처리 테스트
**파일**: `tests/test_error_handling.py`

```python
def test_missing_messages(client):
    """메시지 누락 테스트"""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "phi-3.5-mini",
            "messages": []
        }
    )

    assert response.status_code == 400

def test_invalid_temperature(client, mock_messages):
    """잘못된 temperature 테스트"""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "phi-3.5-mini",
            "messages": mock_messages,
            "temperature": 3.5  # 범위 초과
        }
    )

    assert response.status_code == 422  # Pydantic 검증 오류
```

### 완료 조건
- [ ] 모델 목록 테스트 작성
- [ ] 채팅 완성 테스트 작성
- [ ] 텍스트 완성 테스트 작성
- [ ] 에러 처리 테스트 작성
- [ ] 모든 테스트 통과: `pytest tests/`

---

## Phase 8: 배포 준비

### 목표
Docker 이미지 및 문서를 작성하여 배포 준비를 완료합니다.

### 작업 목록

#### 8.1 Dockerfile 작성
**파일**: `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드
COPY app/ ./app/

# 환경 변수
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# 포트 노출
EXPOSE 8000

# 헬스 체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 8.2 docker-compose.yml 작성
**파일**: `docker-compose.yml`

```yaml
version: '3.8'

services:
  api:
    build: .
    container_name: onnx-genai-api
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/phi-3.5-mini
      - MODEL_ID=phi-3.5-mini
      - SERVER_PORT=8000
      - MAX_CONCURRENT_REQUESTS=10
      - DEFAULT_MAX_TOKENS=1024
      - DEFAULT_TEMPERATURE=0.7
      - CORS_ORIGINS=*
      - LOG_LEVEL=INFO
    volumes:
      - /home/wan/Downloads/models/Phi-3.5-mini-instruct-onnx/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4:/models/phi-3.5-mini:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
```

#### 8.3 README.md 작성
**파일**: `README.md`

```markdown
# ONNX Runtime GenAI OpenAI-Compatible API Server

ONNX Runtime GenAI를 백엔드로 사용하는 OpenAI API 호환 서버입니다.

## 기능

- ✅ OpenAI API 호환 엔드포인트
- ✅ Phi-3.5-mini 모델 지원
- ✅ 스트리밍/논스트리밍 모드
- ✅ 동시 요청 제한
- ✅ 정확한 토큰 사용량 계산
- ✅ 완전한 에러 처리

## 요구사항

- Python 3.10+
- ONNX Runtime GenAI
- Phi-3.5 ONNX 모델

## 설치

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일 편집하여 MODEL_PATH 설정
```

### 3. 서버 실행

```bash
python run.py
```

또는:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker로 실행

```bash
docker-compose up -d
```

## API 사용 예시

### 모델 목록

```bash
curl http://localhost:8000/v1/models
```

### 채팅 완성

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-3.5-mini",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### 스트리밍

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-3.5-mini",
    "messages": [{"role": "user", "content": "Count to 10"}],
    "stream": true
  }'
```

## 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `MODEL_PATH` | 모델 경로 | 필수 |
| `MODEL_ID` | 모델 ID | `phi-3.5-mini` |
| `SERVER_PORT` | 서버 포트 | `8000` |
| `MAX_CONCURRENT_REQUESTS` | 최대 동시 요청 수 | `10` |
| `DEFAULT_MAX_TOKENS` | 기본 max_tokens | `1024` |
| `DEFAULT_TEMPERATURE` | 기본 temperature | `0.7` |
| `CORS_ORIGINS` | CORS 허용 도메인 | `*` |
| `LOG_LEVEL` | 로그 레벨 | `INFO` |

## 테스트

```bash
pytest tests/
```

## 라이선스

MIT
```

#### 8.4 .gitignore 작성
**파일**: `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.pytest_cache/

# Environment
.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
```

### 완료 조건
- [ ] Dockerfile 작성 및 이미지 빌드 테스트
- [ ] docker-compose.yml 작성 및 컨테이너 실행 테스트
- [ ] README.md 작성
- [ ] .gitignore 작성
- [ ] 문서 검토 및 완성도 확인

---

## 전체 구현 체크리스트

### Phase 1: 프로젝트 초기 설정
- [ ] 디렉터리 구조 생성
- [ ] requirements.txt 작성
- [ ] .env.example 작성
- [ ] run.py 작성

### Phase 2: 핵심 인프라
- [ ] settings.py 구현
- [ ] logger.py 구현
- [ ] exceptions.py 구현
- [ ] 스키마 모델 구현 (requests, responses, errors)

### Phase 3: API 엔드포인트 (모의 응답)
- [ ] GET /v1/models 구현
- [ ] POST /v1/chat/completions (모의) 구현
- [ ] POST /v1/completions (모의) 구현
- [ ] FastAPI 앱 통합

### Phase 4: 모델 로더 및 추론
- [ ] prompt_builder.py 구현
- [ ] tokenizer.py 구현
- [ ] model_loader.py 구현
- [ ] inference.py 구현 (논스트리밍)
- [ ] 실제 모델로 추론 테스트

### Phase 5: 스트리밍
- [ ] 채팅 완성 스트리밍 구현
- [ ] 텍스트 완성 스트리밍 구현
- [ ] SSE 형식 검증

### Phase 6: 미들웨어
- [ ] Rate Limiter 구현
- [ ] 로깅 미들웨어 구현
- [ ] 미들웨어 통합 테스트

### Phase 7: 테스트
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 작성
- [ ] 에러 처리 테스트 작성

### Phase 8: 배포
- [ ] Dockerfile 작성
- [ ] docker-compose.yml 작성
- [ ] README.md 작성
- [ ] .gitignore 작성

---

## 다음 단계

구현 계획을 검토한 후, Phase 1부터 순차적으로 구현을 시작하세요. 각 Phase 완료 후 테스트를 수행하여 문제를 조기에 발견하고 수정하는 것이 중요합니다.

구현을 시작하시겠습니까?
