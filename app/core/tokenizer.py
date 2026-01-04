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
