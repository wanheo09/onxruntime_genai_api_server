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
- 모델 로딩 실패 시 표준 에러 형식으로 로그 출력 후 서버 시작 중단
  ```json
  {
    "error": {
      "message": "Failed to load model from /path/to/model: [상세 오류]",
      "type": "server_error",
      "code": "model_loading_failed"
    }
  }
  ```

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

**파라미터 검증**:
- `model`이 `phi-3.5-mini`가 아닌 경우: `invalid_request_error` (code: `model_not_found`)
- `temperature`, `top_p`, `frequency_penalty`, `presence_penalty` 범위 초과 시: `invalid_request_error` (code: `invalid_parameter`, param: 해당 파라미터명)
- `messages` 배열이 비어있거나 형식이 잘못된 경우: `invalid_request_error` (code: `invalid_messages`)

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
- 각 이벤트는 `data: ` 접두사로 시작하고 두 개의 개행 문자(`\n\n`)로 구분
- 스트리밍 진행 중에는 `finish_reason`이 `null`이며, 마지막 청크에서만 실제 값(`stop`, `length` 등)을 포함

```
data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","created":<timestamp>,"model":"phi-3.5-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","created":<timestamp>,"model":"phi-3.5-mini","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","created":<timestamp>,"model":"phi-3.5-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

- 스트리밍 종료는 `data: [DONE]` 메시지로 명시

**기능 요구사항**:
- ONNX Runtime GenAI로 메시지를 적절한 프롬프트 형식으로 변환
- Phi-3.5의 채팅 템플릿 적용 (예: `<|user|>...<|end|><|assistant|>`)
- 스트리밍/논스트리밍 모드 모두 지원
- 토큰 사용량 계산 및 반환:
  - ONNX Runtime GenAI의 토크나이저를 사용하여 정확한 토큰 수 계산
  - `prompt_tokens`: 입력 메시지의 총 토큰 수
  - `completion_tokens`: 생성된 응답의 토큰 수
  - `total_tokens`: prompt_tokens + completion_tokens
- `finish_reason`: `stop` (정상 종료), `length` (max_tokens 도달), `content_filter` (필터링됨) 등

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

**응답 형식 (Streaming)**:
- Content-Type: `text/event-stream`
- SSE (Server-Sent Events) 형식
- `object` 필드는 `text_completion.chunk` 사용

```
data: {"id":"cmpl-<uuid>","object":"text_completion.chunk","created":<timestamp>,"model":"phi-3.5-mini","choices":[{"text":" there","index":0,"finish_reason":null}]}

data: {"id":"cmpl-<uuid>","object":"text_completion.chunk","created":<timestamp>,"model":"phi-3.5-mini","choices":[{"text":" was","index":0,"finish_reason":null}]}

data: {"id":"cmpl-<uuid>","object":"text_completion.chunk","created":<timestamp>,"model":"phi-3.5-mini","choices":[{"text":"","index":0,"finish_reason":"stop"}]}

data: [DONE]
```

**기능 요구사항**:
- 단순 텍스트 완성 (채팅 템플릿 없이)
- 스트리밍/논스트리밍 모드 모두 지원
- 토큰 사용량 계산 (chat/completions와 동일한 방식)

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

**에러 코드 및 HTTP 상태 코드 매핑**:

| 에러 상황 | HTTP 코드 | error.type | error.code | error.param |
|---------|----------|------------|-----------|-------------|
| 존재하지 않는 모델 | 404 | `invalid_request_error` | `model_not_found` | `model` |
| 파라미터 범위 초과 | 400 | `invalid_request_error` | `invalid_parameter` | 해당 파라미터명 |
| 잘못된 messages 형식 | 400 | `invalid_request_error` | `invalid_messages` | `messages` |
| 컨텍스트 길이 초과 | 400 | `invalid_request_error` | `context_length_exceeded` | `messages` 또는 `max_tokens` |
| 필수 파라미터 누락 | 400 | `invalid_request_error` | `missing_parameter` | 누락된 파라미터명 |
| 최대 동시 요청 수 초과 | 429 | `rate_limit_error` | `rate_limit_exceeded` | null |
| 서버 내부 오류 | 500 | `server_error` | `internal_error` | null |
| 모델 로딩 실패 (시작 시) | 500 | `server_error` | `model_loading_failed` | null |

**에러 응답 예시**:
```json
{
  "error": {
    "message": "temperature must be between 0.0 and 2.0, got 3.5",
    "type": "invalid_request_error",
    "param": "temperature",
    "code": "invalid_parameter"
  }
}
```

---

### 5. 비기능 요구사항

#### 5.1 성능

- 동시 요청 처리 지원 (비동기 처리 권장)
- 요청 큐잉 및 타임아웃 설정
- CPU 최적화된 추론

#### 5.2 보안

- 인증 없이 동작 (1차 버전에서는 API 키 인증 미구현)
- CORS 설정 (필요 시 환경 변수로 허용 도메인 설정)
- 요청 크기 제한 (예: 최대 요청 본문 크기 10MB)

#### 5.3 로깅

- 요청/응답 로깅
- 추론 시간 측정 및 로깅
- 에러 로깅

#### 5.4 설정

- 환경 변수로 설정 관리:
  - `MODEL_PATH`: 모델 경로 (기본: `/home/wan/Downloads/models/Phi-3.5-mini-instruct-onnx/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4`)
  - `SERVER_PORT`: 서버 포트 (기본: 8000)
  - `MAX_CONCURRENT_REQUESTS`: 최대 동시 요청 수 (기본: 10)
  - `DEFAULT_MAX_TOKENS`: 기본 max_tokens 값 (기본: 1024)
  - `DEFAULT_TEMPERATURE`: 기본 temperature 값 (기본: 0.7)
  - `CORS_ORIGINS`: CORS 허용 도메인 (기본: "*")
  - `MAX_REQUEST_SIZE_MB`: 최대 요청 크기 (기본: 10)

**동시 요청 수 초과 처리**:
- `MAX_CONCURRENT_REQUESTS`를 초과하는 요청은 429 응답 반환
- 에러 형식:
  ```json
  {
    "error": {
      "message": "Too many concurrent requests. Please try again later.",
      "type": "rate_limit_error",
      "code": "rate_limit_exceeded"
    }
  }
  ```

---

### 6. 제약사항 및 고려사항

- ONNX Runtime GenAI의 CPU 백엔드 성능 제약
- Phi-3.5 모델의 컨텍스트 길이 제한 확인 필요
- 멀티모달 기능 (이미지 등)은 구현 범위에서 제외
- 임베딩 API는 구현하지 않음 (Phi-3.5가 텍스트 생성 모델이므로)

---

### 7. 구현 범위

모든 기능이 동등한 중요도를 가지며, 다음 항목들이 모두 구현되어야 합니다:

1. **GET /v1/models** - 모델 목록 조회
2. **POST /v1/chat/completions** - 채팅 완성 (streaming/non-streaming)
3. **POST /v1/completions** - 텍스트 완성 (streaming/non-streaming)
4. **완전한 에러 처리** - 모든 에러 케이스에 대한 표준 응답 형식
5. **비기능 요구사항**:
   - 비동기 요청 처리
   - 동시 요청 수 제한 및 429 응답
   - CORS 설정
   - 환경 변수 기반 설정
   - 로깅 (요청/응답, 추론 시간, 에러)
6. **토큰 사용량 정확한 계산** - ONNX Runtime GenAI 토크나이저 사용

**구현하지 않는 기능**:
- POST /v1/embeddings (임베딩 API)
- API 키 인증
- 멀티모달 기능
