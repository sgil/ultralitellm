from __future__ import annotations

import pytest

from ultralitellm.errors import AllProvidersFailed, ModelNotFound
from ultralitellm.providers import FailingProvider, FlakyProvider, StaticProvider
from ultralitellm.router import Router
from ultralitellm.schemas import ChatCompletionRequest


def _request(model: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[{"role": "user", "content": "hi"}])


def _router() -> Router:
    providers = {
        "primary": StaticProvider("primary", "from primary"),
        "fallback": StaticProvider("fallback", "from fallback"),
        "broken": FailingProvider("broken"),
    }
    routes = {
        "ok": ["primary", "fallback"],
        "fb": ["broken", "fallback"],
        "dead": ["broken", "broken"],
    }
    return Router(providers, routes)


def test_primary_serves_when_available() -> None:
    outcome = _router().complete(_request("ok"))
    assert outcome.provider == "primary"
    assert outcome.result.content == "from primary"
    assert outcome.attempts == []


def test_falls_back_when_primary_fails() -> None:
    outcome = _router().complete(_request("fb"))
    assert outcome.provider == "fallback"
    assert outcome.result.content == "from fallback"
    assert [name for name, _ in outcome.attempts] == ["broken"]


def test_all_providers_failed_raises_with_attempts() -> None:
    with pytest.raises(AllProvidersFailed) as info:
        _router().complete(_request("dead"))
    assert info.value.model == "dead"
    assert len(info.value.attempts) == 2


def test_unknown_model_raises_model_not_found() -> None:
    with pytest.raises(ModelNotFound):
        _router().complete(_request("nope"))


def test_chain_order_is_respected() -> None:
    providers = {
        "a": FlakyProvider("a", fail_times=1, content="a-ok"),
        "b": StaticProvider("b", "b-ok"),
        "c": StaticProvider("c", "c-ok"),
    }
    router = Router(providers, {"m": ["a", "b", "c"]})
    first = router.complete(_request("m"))
    assert first.provider == "b"
    second = router.complete(_request("m"))
    assert second.provider == "a"
    assert second.result.content == "a-ok"


def test_empty_chain_rejected() -> None:
    with pytest.raises(ValueError):
        Router({"p": StaticProvider("p", "x")}, {"m": []})


def test_unknown_provider_reference_rejected() -> None:
    with pytest.raises(ValueError):
        Router({"p": StaticProvider("p", "x")}, {"m": ["ghost"]})
