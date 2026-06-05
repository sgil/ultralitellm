from __future__ import annotations

from fastapi.testclient import TestClient


def test_successful_completion_uses_chat_completion_shape(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "support-chat", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert isinstance(body["created"], int)

    assert body["model"] == "support-chat"
    assert body["provider"] == "primary"

    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Hi! How can I help?"

    assert choice["finish_reason"] == "stop"

    assert resp.headers["X-UltraLiteLLM-Provider"] == "primary"


def test_fallback_keeps_same_shape_and_names_real_provider(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "flaky-chat", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "fallback"
    assert body["model"] == "flaky-chat"
    assert body["choices"][0]["message"]["content"] == "Backup here."
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert resp.headers["X-UltraLiteLLM-Provider"] == "fallback"


def test_unknown_model_returns_404_error_envelope(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "does-not-exist", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "model_not_found"
    assert error["type"] == "invalid_request_error"


def test_all_providers_failed_returns_502(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "doomed", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 502
    error = resp.json()["error"]
    assert error["code"] == "all_providers_failed"


def test_echo_model_reflects_user_message(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "echo-model", "messages": [{"role": "user", "content": "ping"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ping"


def test_empty_messages_is_rejected(client: TestClient) -> None:
    resp = client.post("/v1/chat/completions", json={"model": "support-chat", "messages": []})
    assert resp.status_code == 422


def test_missing_model_is_rejected(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 422


def test_list_models(client: TestClient) -> None:
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    ids = {m["id"] for m in body["data"]}
    assert {"support-chat", "flaky-chat", "echo-model"} <= ids
