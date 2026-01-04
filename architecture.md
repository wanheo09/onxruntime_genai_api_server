# ONNX Runtime GenAI OpenAI‑Compatible API 서버 아키텍처

## 전체 구성
- **FastAPI** 를 메인 웹 프레임워크로 사용하고, **uvicorn** 으로 ASGI 서버를 실행합니다.
- 모든 설정은 환경 변수(`MODEL_PATH`, `SERVER_PORT` 등) 로 관리하며, `settings.py` 에 단일 설정 객체를 제공합니다.

## 핵심 모듈
1. **ModelLoader**
   - 서버 시작 시 `MODEL_PATH` 로 지정된 모델을 **ONNX Runtime GenAI**(CPU EP, INT4 AWQ) 로 로드합니다.
   - 로드 실패 시 `model_loading_failed` 오류를 반환하고 서버를 종료합니다.
   - 싱글톤 형태로 애플리케이션 전역에서 공유됩니다.
2. **Tokenizer**
   - 모델에 포함된 토크나이저를 이용해 `prompt_tokens`, `completion_tokens`, `total_tokens` 를 정확히 계산합니다.
3. **InferenceService**
   - `chat/completions` 와 `completions` 엔드포인트 모두에서 사용됩니다.
   - 입력 메시지를 **Phi‑3.5** 전용 템플릿(`"<|user|>...<|assistant|>"`) 으로 변환 후 `session.run` 으로 추론합니다.
   - `stream` 파라미터에 따라 **async generator** 로 SSE(`text/event-stream`) 형태를 반환합니다.
   - `max_tokens`, `temperature`, `top_p` 등 파라미터 검증 로직을 포함합니다.
4. **RateLimiter Middleware**
   - `MAX_CONCURRENT_REQUESTS` 를 초과하는 요청에 대해 429 응답(`rate_limit_exceeded`)을 반환합니다.
5. **CORS Middleware**
   - `CORS_ORIGINS` 환경 변수(기본 `*`) 로 허용 도메인을 설정합니다.

## 엔드포인트 정의
- `GET /v1/models`
  - 고정된 모델 목록(`phi-3.5-mini`)을 반환합니다. `created` 타임스탬프는 서버 시작 시生成됩니다.
- `POST /v1/chat/completions`
  - **ChatCompletionRequest** 를 검증하고 `InferenceService` 를 호출합니다.
  - `stream` 값에 따라 비스트리밍 혹은 스트리밍 응답을 제공합니다.
- `POST /v1/completions`
  - **TextCompletionRequest** 를 검증하고 동일 서비스(`InferenceService`)를 사용합니다.
  - 템플릿 없이 순수 텍스트 생성합니다.

## 에러 처리 흐름
- **Pydantic** 모델 검증 → 파라미터 누락·형식 오류 → `invalid_request_error`(code `missing_parameter`/`invalid_parameter`).
- 비즈니스 로직 검증 (model ID, 파라미터 범위, messages 형식 등) → 해당 `error.type`, `error.code`, `error.param` 포함 응답.
- 내부 예외(추론 실패 등) → `server_error`(code `internal_error`) 로 반환.

## 로깅 & 모니터링 (`logger.py`)
- 요청/응답 메타데이터(URI, 파라미터, 상태 코드) 기록
- 추론 시작·종료 시점 시간 차를 `inference_time_ms` 로 로깅
- 에러 발생 시 스택 트레이스와 함께 `error` 로그 기록

## 비동기·동시성
- FastAPI‑async 엔드포인트와 `asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)` 로 동시 요청을 제한합니다.
- CPU 바인드 추론 작업은 필요 시 `ThreadPoolExecutor` 로 오프로드할 수 있습니다.

## 배포·운영
- **Dockerfile** (베이스: `python:3.10-slim`)
  - `onnxruntime` 및 `onnxruntime-genai` 설치
  - `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "$SERVER_PORT"]`
- 환경 변수 파일 `.env` 로 설정 관리, Kubernetes ConfigMap/Secret 연동 가능.

## 확장 포인트
- 현재 인증(API‑Key) 미구현 → 추후 `AuthMiddleware` 추가 가능
- 멀티모달, 임베딩 API 등은 별도 라우터와 서비스 레이어로 확장 가능

---

위 설계는 `requirements.md` 에 명시된 모든 기능·비기능 요구사항을 충족하도록 구성되었습니다. 실제 구현 시 각 모듈을 `app/` 패키지 아래에 배치하고, `settings.py`, `logger.py`, `model.py`, `inference.py`, `routes/*.py` 로 구분하면 유지보수가 용이합니다.
