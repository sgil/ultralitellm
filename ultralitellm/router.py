from __future__ import annotations

from dataclasses import dataclass, field

from .errors import AllProvidersFailed, ModelNotFound, ProviderError
from .providers.base import Provider, ProviderResult
from .schemas import ChatCompletionRequest


@dataclass
class CompletionOutcome:
    provider: str
    result: ProviderResult
    attempts: list[tuple[str, ProviderError]] = field(default_factory=list)


class Router:
    def __init__(
        self,
        providers: dict[str, Provider],
        routes: dict[str, list[str]],
    ) -> None:
        self.providers = providers
        self.routes = routes
        self._validate()

    def _validate(self) -> None:
        for model, chain in self.routes.items():
            if not chain:
                raise ValueError(f"model {model!r} has an empty provider chain")
            for name in chain:
                if name not in self.providers:
                    raise ValueError(
                        f"model {model!r} references unknown provider {name!r}"
                    )

    @property
    def models(self) -> list[str]:
        return list(self.routes)

    def chain_for(self, model: str) -> list[str]:
        try:
            return self.routes[model]
        except KeyError as exc:
            raise ModelNotFound(model) from exc

    def complete(self, request: ChatCompletionRequest) -> CompletionOutcome:
        chain = self.chain_for(request.model)
        attempts: list[tuple[str, ProviderError]] = []
        for name in chain:
            provider = self.providers[name]
            try:
                result = provider.complete(request)
            except ProviderError as exc:
                attempts.append((name, exc))
                continue
            return CompletionOutcome(provider=name, result=result, attempts=attempts)
        raise AllProvidersFailed(request.model, attempts)
