"""Together serverless: nothing to start or stop; billed per input and output token."""

from __future__ import annotations

import os

import httpx

from providers.base import Endpoint, Provider


class Together(Provider):
    name = "together"
    self_hosted = False

    def __init__(self, cfg: dict, model_id: str):
        self.cfg = cfg
        self.model_id = model_id

    def _headers(self) -> dict:
        key = os.environ.get("TOGETHER_API_KEY")
        if not key:
            raise SystemExit("TOGETHER_API_KEY is not set. Put it in .env (see .env.example).")
        return {"Authorization": f"Bearer {key}"}

    def list_models(self) -> list[dict]:
        r = httpx.get(f"{self.cfg['base_url']}/models", headers=self._headers(), timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["data"] if isinstance(data, dict) else data

    def check_model(self) -> dict:
        """Stop before any paid request if the configured model id is not served."""
        models = self.list_models()
        hit = [m for m in models if m.get("id") == self.model_id]
        if not hit:
            near = sorted(m["id"] for m in models if "qwen" in m.get("id", "").lower())
            raise SystemExit(f"Together does not list {self.model_id!r}. Qwen models it lists: {near}")
        return hit[0]

    def start(self, variant: str = "serverless") -> Endpoint:
        return Endpoint(
            provider=self.name,
            variant=variant,
            base_url=self.cfg["base_url"],
            model=self.model_id,
            headers=self._headers(),
            extra_body=dict(self.cfg.get("extra_body") or {}),
        )

    def run_cost_usd(self, rows: list[dict], billed_seconds: float = 0.0) -> float:
        p_in = sum(r.get("prompt_tokens") or 0 for r in rows)
        p_out = sum(r.get("completion_tokens") or 0 for r in rows)
        return (p_in * self.cfg["usd_per_m_input"] + p_out * self.cfg["usd_per_m_output"]) / 1e6

    def usd_per_1k_requests(self, prompt_tokens: float, completion_tokens: float) -> float:
        """Posted price for 1,000 requests at the measured mean input and output lengths."""
        return 1000 * (prompt_tokens * self.cfg["usd_per_m_input"]
                       + completion_tokens * self.cfg["usd_per_m_output"]) / 1e6
