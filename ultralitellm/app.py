from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import load_router
from .errors import AllProvidersFailed, ModelNotFound
from .router import CompletionOutcome, Router
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
)


def _generate_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _error_response(status_code: int, message: str, error_type: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "code": code}},
    )


def _build_response(
    request: ChatCompletionRequest, outcome: CompletionOutcome
) -> ChatCompletionResponse:
    result = outcome.result
    return ChatCompletionResponse(
        id=_generate_id(),
        created=int(time.time()),
        model=request.model,
        provider=outcome.provider,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role=result.role, content=result.content),
                finish_reason=result.finish_reason,
            )
        ],
        usage=result.usage,
    )


def create_app(router: Router | None = None) -> FastAPI:
    app = FastAPI(title="UltraLiteLLM", version="0.1.0")
    app.state.router = router or load_router()

    @app.exception_handler(ModelNotFound)
    async def _model_not_found(_: Request, exc: ModelNotFound) -> JSONResponse:
        return _error_response(404, str(exc), "invalid_request_error", "model_not_found")

    @app.exception_handler(AllProvidersFailed)
    async def _all_failed(_: Request, exc: AllProvidersFailed) -> JSONResponse:
        return _error_response(502, str(exc), "upstream_error", "all_providers_failed")

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        models = app.state.router.models
        return {
            "object": "list",
            "data": [
                {"id": model, "object": "model", "owned_by": "ultralitellm"}
                for model in models
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> JSONResponse:
        outcome = app.state.router.complete(request)
        response = _build_response(request, outcome)
        return JSONResponse(
            content=response.model_dump(exclude_none=True),
            headers={"X-UltraLiteLLM-Provider": outcome.provider},
        )

    return app


app = create_app()
