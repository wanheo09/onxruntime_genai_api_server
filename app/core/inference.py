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

            # finish_reason 결정
            finish_reason = "length" if completion_tokens >= max_tokens else "stop"

            return {
                "content": generated_text,
                "finish_reason": finish_reason,
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

            # finish_reason 결정
            finish_reason = "length" if completion_tokens >= max_tokens else "stop"

            # 마지막 청크 (finish_reason 포함)
            yield {
                "delta": {},
                "finish_reason": finish_reason,
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
