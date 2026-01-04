from typing import Optional


class APIException(Exception):
    """기본 API 예외 클래스"""
    def __init__(
        self,
        message: str,
        type: str,
        code: str,
        param: Optional[str] = None,
        status_code: int = 400
    ):
        self.message = message
        self.type = type
        self.code = code
        self.param = param
        self.status_code = status_code
        super().__init__(self.message)


class ModelNotFoundError(APIException):
    """모델을 찾을 수 없음"""
    def __init__(self, model: str):
        super().__init__(
            message=f"Model '{model}' not found",
            type="invalid_request_error",
            code="model_not_found",
            param="model",
            status_code=404
        )


class InvalidParameterError(APIException):
    """잘못된 파라미터"""
    def __init__(self, param: str, message: str):
        super().__init__(
            message=message,
            type="invalid_request_error",
            code="invalid_parameter",
            param=param,
            status_code=400
        )


class InvalidMessagesError(APIException):
    """잘못된 메시지 형식"""
    def __init__(self, message: str = "Invalid messages format"):
        super().__init__(
            message=message,
            type="invalid_request_error",
            code="invalid_messages",
            param="messages",
            status_code=400
        )


class ContextLengthExceededError(APIException):
    """컨텍스트 길이 초과"""
    def __init__(self, param: str = "messages"):
        super().__init__(
            message="Context length exceeded",
            type="invalid_request_error",
            code="context_length_exceeded",
            param=param,
            status_code=400
        )


class MissingParameterError(APIException):
    """필수 파라미터 누락"""
    def __init__(self, param: str):
        super().__init__(
            message=f"Missing required parameter: {param}",
            type="invalid_request_error",
            code="missing_parameter",
            param=param,
            status_code=400
        )


class RateLimitExceededError(APIException):
    """요청 제한 초과"""
    def __init__(self):
        super().__init__(
            message="Too many concurrent requests. Please try again later.",
            type="rate_limit_error",
            code="rate_limit_exceeded",
            status_code=429
        )


class ModelLoadingError(APIException):
    """모델 로딩 실패"""
    def __init__(self, path: str, detail: str):
        super().__init__(
            message=f"Failed to load model from {path}: {detail}",
            type="server_error",
            code="model_loading_failed",
            status_code=500
        )


class InternalServerError(APIException):
    """서버 내부 오류"""
    def __init__(self, message: str = "Internal server error"):
        super().__init__(
            message=message,
            type="server_error",
            code="internal_error",
            status_code=500
        )
