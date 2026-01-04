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
