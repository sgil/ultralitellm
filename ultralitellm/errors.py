from __future__ import annotations


class ProviderError(Exception):
    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"provider {provider!r} failed: {message}")


class ModelNotFound(Exception):
    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"no providers configured for model {model!r}")


class AllProvidersFailed(Exception):
    def __init__(self, model: str, attempts: list[tuple[str, ProviderError]]) -> None:
        self.model = model
        self.attempts = attempts
        chain = ", ".join(f"{name}: {err}" for name, err in attempts)
        super().__init__(f"all providers failed for model {model!r} ({chain})")
