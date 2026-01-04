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

### 텍스트 완성

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-3.5-mini",
    "prompt": "Once upon a time",
    "max_tokens": 50
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

## API 엔드포인트

### GET /v1/models

사용 가능한 모델 목록을 반환합니다.

### POST /v1/chat/completions

채팅 완성 API입니다. 스트리밍 및 논스트리밍을 모두 지원합니다.

**파라미터:**
- `model` (required): 모델 ID
- `messages` (required): 메시지 배열
- `temperature` (optional): 0.0 ~ 2.0 (기본값: 0.7)
- `max_tokens` (optional): 최대 생성 토큰 수 (기본값: 1024)
- `top_p` (optional): 0.0 ~ 1.0 (기본값: 1.0)
- `stream` (optional): 스트리밍 여부 (기본값: false)
- `stop` (optional): 정지 시퀀스

### POST /v1/completions

텍스트 완성 API입니다. 스트리밍 및 논스트리밍을 모두 지원합니다.

**파라미터:**
- `model` (required): 모델 ID
- `prompt` (required): 입력 프롬프트
- `temperature` (optional): 0.0 ~ 2.0 (기본값: 0.7)
- `max_tokens` (optional): 최대 생성 토큰 수 (기본값: 100)
- `top_p` (optional): 0.0 ~ 1.0 (기본값: 1.0)
- `stream` (optional): 스트리밍 여부 (기본값: false)

## 에러 처리

모든 에러는 OpenAI API 호환 형식으로 반환됩니다:

```json
{
  "error": {
    "message": "Error message",
    "type": "error_type",
    "code": "error_code",
    "param": "parameter_name"
  }
}
```

## 테스트

```bash
pytest tests/
```

## 프로젝트 구조

```
onxruntime_genai_api_server/
├── app/
│   ├── config/          # 설정
│   ├── models/          # Pydantic 스키마
│   ├── core/            # 핵심 로직 (모델, 추론)
│   ├── middleware/      # 미들웨어
│   ├── routes/          # API 라우터
│   └── utils/           # 유틸리티
├── tests/               # 테스트
├── requirements.txt     # 의존성
└── run.py              # 실행 스크립트
```

## 라이선스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 참고 문서

- [requirements.md](requirements.md) - 요구사항 명세서
- [design.md](design.md) - 상세 설계 문서
- [implementation-plan.md](implementation-plan.md) - 구현 계획
