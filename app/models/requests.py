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
