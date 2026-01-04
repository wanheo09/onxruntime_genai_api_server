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
