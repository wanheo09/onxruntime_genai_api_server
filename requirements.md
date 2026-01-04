# ONNX Runtime GenAI - OpenAI Compatible API Server
## 요구사항 명세서

### 1. 프로젝트 개요

**목적**: ONNX Runtime GenAI를 백엔드로 사용하여 OpenAI API와 호환되는 REST API 서버 구현

**기술 스택**:
- 언어: Python
- 추론 엔진: ONNX Runtime GenAI (CPU backend)
- API 프레임워크: FastAPI (권장) 또는 Flask

---

### 2. 모델 설정

#### 2.1 모델 정보

- **모델 ID**: `phi-3.5-mini`
- **모델 경로**: `/home/wan/Downloads/models/Phi-3.5-mini-instruct-onnx/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4`
- **백엔드**: CPU (ONNX Runtime CPU Execution Provider)
- **양자화**: INT4 AWQ (block-128, accuracy level 4)

#### 2.2 모델 로딩

- 서버 시작 시 모델 사전 로드
- 모델 로딩 실패 시 명확한 에러 메시지 출력 및 서버 시작 중단

---

### 3. API 엔드포인트 명세

#### 3.1 GET /v1/models

**기능**: 사용 가능한 모델 목록 반환

**응답 형식**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "phi-3.5-mini",
      "object": "model",
      "created": <timestamp>,
      "owned_by": "system"
    }
  ]
}
```

**요구사항**:
- 항상 단일 모델 (`phi-3.5-mini`) 반환
- OpenAI API 응답 형식과 동일한 구조

---

#### 3.2 POST /v1/chat/completions

**기능**: 채팅 완성 API (주요 엔드포인트)

**요청 형식**:
```json
{
  "model": "phi-3.5-mini",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "top_p": 1.0,
  "stream": false,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0
}
```

**필수 파라미터**:
- `model`: 모델 ID (검증 필요)
- `messages`: 채팅 메시지 배열

**선택 파라미터**:
- `temperature`: 0.0 ~ 2.0 (기본값: 0.7)
- `max_tokens`: 최대 생성 토큰 수 (기본값: 1024)
- `top_p`: 0.0 ~ 1.0 (기본값: 1.0)
- `stream`: 스트리밍 여부 (기본값: false)
- `frequency_penalty`: 0.0 ~ 2.0 (기본값: 0.0)
- `presence_penalty`: 0.0 ~ 2.0 (기본값: 0.0)
- `stop`: 정지 시퀀스 (문자열 또는 배열)

**응답 형식 (Non-streaming)**:
```json
{
  "id": "chatcmpl-<uuid>",
  "object": "chat.completion",
  "created": <timestamp>,
  "model": "phi-3.5-mini",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
  }
}
```

**응답 형식 (Streaming)**:
- Content-Type: `text/event-stream`
- SSE (Server-Sent Events) 형식

```
data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","created":<timestamp>,"model":"phi-3.5-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","created":<timestamp>,"model":"phi-3.5-mini","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","created":<timestamp>,"model":"phi-3.5-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**기능 요구사항**:
- ONNX Runtime GenAI로 메시지를 적절한 프롬프트 형식으로 변환
- Phi-3.5의 채팅 템플릿 적용 (예: `<|user|>...<|end|><|assistant|>`)
- 스트리밍/논스트리밍 모드 모두 지원
- 토큰 사용량 계산 및 반환
- `finish_reason`: `stop`, `length`, `content_filter` 등

---

#### 3.3 POST /v1/completions

**기능**: 텍스트 완성 API (레거시)

**요청 형식**:
```json
{
  "model": "phi-3.5-mini",
  "prompt": "Once upon a time",
  "temperature": 0.7,
  "max_tokens": 100,
  "top_p": 1.0,
  "stream": false
}
```

**응답 형식**:
```json
{
  "id": "cmpl-<uuid>",
  "object": "text_completion",
  "created": <timestamp>,
  "model": "phi-3.5-mini",
  "choices": [
    {
      "text": " there was a kingdom...",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 15,
    "total_tokens": 20
  }
}
```

**기능 요구사항**:
- 단순 텍스트 완성 (채팅 템플릿 없이)
- 스트리밍 지원 (chat/completions와 유사)

---

#### 3.4 POST /v1/embeddings

**기능**: 텍스트 임베딩 생성

**요청 형식**:
```json
{
  "model": "phi-3.5-mini",
  "input": "The quick brown fox jumps over the lazy dog",
  "encoding_format": "float"
}
```

**응답 형식**:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, -0.3, ...],
      "index": 0
    }
  ],
  "model": "phi-3.5-mini",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

**기능 요구사항**:
- `input`: 문자열 또는 문자열 배열 지원
- `encoding_format`: `float` (기본값) 또는 `base64`
- ONNX Runtime GenAI로 텍스트 임베딩 생성
- **참고**: Phi-3.5가 임베딩 전용 모델이 아닐 수 있음 - 마지막 hidden state 사용 또는 별도 처리 필요

---

### 4. 에러 처리

**에러 응답 형식**:
```json
{
  "error": {
    "message": "Invalid model specified",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

**주요 에러 케이스**:
- 존재하지 않는 모델 ID 요청
- 잘못된 파라미터 값 (범위 초과 등)
- 최대 토큰 길이 초과
- 서버 내부 오류

**HTTP 상태 코드**:
- 400: Bad Request (잘못된 요청)
- 404: Not Found (모델 없음)
- 500: Internal Server Error (서버 오류)

---

### 5. 비기능 요구사항

#### 5.1 성능

- 동시 요청 처리 지원 (비동기 처리 권장)
- 요청 큐잉 및 타임아웃 설정
- CPU 최적화된 추론

#### 5.2 보안

- API 키 인증 (선택 사항, OpenAI 호환성 위해)
- CORS 설정
- 요청 크기 제한

#### 5.3 로깅

- 요청/응답 로깅
- 추론 시간 측정 및 로깅
- 에러 로깅

#### 5.4 설정

- 환경 변수로 설정 관리:
  - 모델 경로
  - 서버 포트 (기본: 8000)
  - 최대 동시 요청 수
  - 기본 파라미터 값

---

### 6. 제약사항 및 고려사항

- ONNX Runtime GenAI의 CPU 백엔드 성능 제약
- Phi-3.5 모델의 컨텍스트 길이 제한 확인 필요
- 임베딩 API는 Phi-3.5가 채팅 모델이므로 제한적일 수 있음
- 멀티모달 기능 (이미지 등)은 1차 버전에서 제외

---

### 7. 구현 범위

모든 기능이 동등한 중요도를 가지며, 다음 항목들이 모두 구현되어야 합니다:

1. GET /v1/models
2. POST /v1/chat/completions (streaming/non-streaming)
3. POST /v1/completions (streaming/non-streaming)
4. POST /v1/embeddings
5. 완전한 에러 처리
6. 비기능 요구사항 (성능, 보안, 로깅, 설정)
