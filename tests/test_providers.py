from __future__ import annotations

import pytest

from ultralitellm.errors import ProviderError
from ultralitellm.providers import (
    EchoProvider,
    FailingProvider,
    FlakyProvider,
    StaticProvider,
)
from ultralitellm.schemas import ChatCompletionRequest


def _request(content: str = "Hello", model: str = "m") -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[{"role": "user", "content": content}])


def test_static_provider_returns_fixed_content() -> None:
    provider = StaticProvider("primary", "fixed reply")
    result = provider.complete(_request())
    assert result.content == "fixed reply"
    assert result.role == "assistant"
    assert result.finish_reason == "stop"


def test_echo_provider_reflects_last_user_message() -> None:
    provider = EchoProvider("echo", prefix="echo: ")
    req = ChatCompletionRequest(
        model="m",
        messages=[
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "second"},
        ],
    )
    assert provider.complete(req).content == "echo: second"


def test_failing_provider_raises_provider_error() -> None:
    provider = FailingProvider("broken", status_code=503)
    with pytest.raises(ProviderError) as info:
        provider.complete(_request())
    assert info.value.provider == "broken"
    assert info.value.status_code == 503


def test_flaky_provider_fails_then_succeeds() -> None:
    provider = FlakyProvider("flaky", fail_times=2, content="recovered")
    with pytest.raises(ProviderError):
        provider.complete(_request())
    with pytest.raises(ProviderError):
        provider.complete(_request())
    assert provider.complete(_request()).content == "recovered"
