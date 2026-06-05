from __future__ import annotations

import abc
from dataclasses import dataclass

from ..schemas import ChatCompletionRequest, Usage


@dataclass
class ProviderResult:
    content: str
    finish_reason: str = "stop"
    role: str = "assistant"
    usage: Usage | None = None


class Provider(abc.ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def complete(self, request: ChatCompletionRequest) -> ProviderResult:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
