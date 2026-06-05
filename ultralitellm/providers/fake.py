from __future__ import annotations

from ..errors import ProviderError
from ..schemas import ChatCompletionRequest, Usage
from .base import Provider, ProviderResult


def _last_user_content(request: ChatCompletionRequest) -> str:
    for message in reversed(request.messages):
        if message.role == "user":
            return message.content
    return request.messages[-1].content if request.messages else ""


class StaticProvider(Provider):
    def __init__(self, name: str, content: str, *, finish_reason: str = "stop") -> None:
        super().__init__(name)
        self.content = content
        self.finish_reason = finish_reason

    def complete(self, request: ChatCompletionRequest) -> ProviderResult:
        return ProviderResult(
            content=self.content,
            finish_reason=self.finish_reason,
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )


class EchoProvider(Provider):
    def __init__(self, name: str, *, prefix: str = "") -> None:
        super().__init__(name)
        self.prefix = prefix

    def complete(self, request: ChatCompletionRequest) -> ProviderResult:
        content = f"{self.prefix}{_last_user_content(request)}"
        return ProviderResult(content=content, finish_reason="stop")


class FailingProvider(Provider):
    def __init__(
        self,
        name: str,
        *,
        message: str = "simulated upstream failure",
        status_code: int | None = 500,
    ) -> None:
        super().__init__(name)
        self.message = message
        self.status_code = status_code

    def complete(self, request: ChatCompletionRequest) -> ProviderResult:
        raise ProviderError(self.name, self.message, status_code=self.status_code)


class FlakyProvider(Provider):
    def __init__(
        self,
        name: str,
        *,
        fail_times: int,
        content: str = "recovered",
        message: str = "transient failure",
    ) -> None:
        super().__init__(name)
        self.remaining_failures = fail_times
        self.content = content
        self.message = message

    def complete(self, request: ChatCompletionRequest) -> ProviderResult:
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise ProviderError(self.name, self.message, status_code=503)
        return ProviderResult(content=self.content, finish_reason="stop")
