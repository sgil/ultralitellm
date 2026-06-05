from __future__ import annotations

import json

import pytest

from ultralitellm.config import build_router, default_config, load_router
from ultralitellm.schemas import ChatCompletionRequest


def test_default_config_builds_a_working_router() -> None:
    router = build_router(default_config())
    assert set(router.models) >= {"support-chat", "flaky-chat", "echo-model"}

    req = ChatCompletionRequest(
        model="support-chat", messages=[{"role": "user", "content": "hi"}]
    )
    assert router.complete(req).provider == "primary"


def test_load_router_reads_config_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "providers": {"only": {"type": "static", "content": "from file"}},
        "routes": {"m": ["only"]},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("ULTRALITELLM_CONFIG", str(path))

    router = load_router()
    req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "x"}])
    outcome = router.complete(req)
    assert outcome.provider == "only"
    assert outcome.result.content == "from file"


def test_unknown_provider_type_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_router({"providers": {"x": {"type": "bogus"}}, "routes": {}})
