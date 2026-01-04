from pydantic import BaseModel
from typing import Optional


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
