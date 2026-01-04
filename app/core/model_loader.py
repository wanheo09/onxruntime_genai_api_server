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
