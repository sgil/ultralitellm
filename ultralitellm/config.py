from __future__ import annotations

import json
import os
from typing import Any

from .providers import (
    EchoProvider,
    FailingProvider,
    FlakyProvider,
    Provider,
    StaticProvider,
)
from .router import Router

CONFIG_ENV_VAR = "ULTRALITELLM_CONFIG"


def default_config() -> dict[str, Any]:
    return {
        "providers": {
            "primary": {"type": "static", "content": "Hi! How can I help?"},
            "fallback": {
                "type": "static",
                "content": "Hello from the backup provider.",
            },
            "echo": {"type": "echo"},
            "unstable": {"type": "failing"},
        },
        "routes": {
            "support-chat": ["primary", "fallback"],
            "flaky-chat": ["unstable", "fallback"],
            "echo-model": ["echo"],
        },
    }


def _build_provider(name: str, spec: dict[str, Any]) -> Provider:
    kind = spec.get("type")
    if kind == "static":
        return StaticProvider(
            name,
            content=spec.get("content", "ok"),
            finish_reason=spec.get("finish_reason", "stop"),
        )
    if kind == "echo":
        return EchoProvider(name, prefix=spec.get("prefix", ""))
    if kind == "failing":
        return FailingProvider(
            name,
            message=spec.get("message", "simulated upstream failure"),
            status_code=spec.get("status_code", 500),
        )
    if kind == "flaky":
        return FlakyProvider(
            name,
            fail_times=spec.get("fail_times", 1),
            content=spec.get("content", "recovered"),
        )
    raise ValueError(f"unknown provider type {kind!r} for provider {name!r}")


def build_router(config: dict[str, Any]) -> Router:
    providers = {
        name: _build_provider(name, spec)
        for name, spec in config.get("providers", {}).items()
    }
    routes = {model: list(chain) for model, chain in config.get("routes", {}).items()}
    return Router(providers, routes)


def load_router() -> Router:
    path = os.environ.get(CONFIG_ENV_VAR)
    if path:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    else:
        config = default_config()
    return build_router(config)
