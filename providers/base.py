"""What the benchmark needs from a provider: an OpenAI-compatible endpoint, and a way to price a run."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Endpoint:
    provider: str
    variant: str
    base_url: str                 # ends in /v1
    model: str
    headers: dict[str, str]
    extra_body: dict = field(default_factory=dict)   # provider-only request fields
    metrics_url: str | None = None                   # Prometheus /metrics, self-hosted only
    honours_min_tokens: bool = False                 # vLLM does; then every request generates the full output
    shared_fleet: bool = False                       # serverless: past a point we measure the rate limit, not a GPU


class Provider:
    name: str = "base"
    self_hosted: bool = False

    def start(self, variant: str) -> Endpoint:
        raise NotImplementedError

    def stop(self, variant: str) -> None:
        """Release anything that bills while idle. Serverless providers do nothing."""

    def run_cost_usd(self, rows: list[dict], billed_seconds: float) -> float:
        """Spend for a finished phase: tokens for serverless, container-seconds for self-hosted."""
        raise NotImplementedError
