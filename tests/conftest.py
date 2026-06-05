from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ultralitellm.app import create_app
from ultralitellm.providers import (
    EchoProvider,
    FailingProvider,
    StaticProvider,
)
from ultralitellm.router import Router


@pytest.fixture
def router() -> Router:
    providers = {
        "primary": StaticProvider("primary", "Hi! How can I help?"),
        "fallback": StaticProvider("fallback", "Backup here."),
        "broken": FailingProvider("broken"),
        "echo": EchoProvider("echo"),
    }
    routes = {
        "support-chat": ["primary", "fallback"],
        "flaky-chat": ["broken", "fallback"],
        "doomed": ["broken", "broken"],
        "echo-model": ["echo"],
    }
    return Router(providers, routes)


@pytest.fixture
def client(router: Router) -> TestClient:
    return TestClient(create_app(router))
