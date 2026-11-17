"""Self-hosted vLLM on Modal: deploy an app per variant, find its URL, stop it and confirm it stopped."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from providers.base import Endpoint, Provider

APP_FILE = Path(__file__).with_name("modal_app.py")


def app_name(variant: str) -> str:
    return f"inference-poc-{variant}"


def running_poc_apps() -> list[str]:
    """Names of inference-poc apps Modal does not yet list as stopped (state "stopping..." counts: it can still hold a container)."""
    r = subprocess.run(["modal", "app", "list", "--json"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"modal app list failed: {r.stderr.strip()}")
    apps = json.loads(r.stdout or "[]")
    out = []
    for a in apps:
        name = a.get("Description") or a.get("description") or a.get("name") or ""
        state = (a.get("State") or a.get("state") or "").lower().rstrip(".")
        if name.startswith("inference-poc-") and state != "stopped":
            out.append(f"{name} ({state})")
    return out


class ModalVLLM(Provider):
    name = "modal_vllm"
    self_hosted = True

    def __init__(self, cfg: dict, served_name: str):
        self.cfg = cfg
        self.served_name = served_name

    # Settings a variant may override; anything it leaves out comes from the provider block.
    PER_VARIANT = ("gpu", "cpu_cores", "memory_gib", "usd_per_sec_gpu", "vllm", "transformers",
                   "max_model_len", "served_name", "min_tokens")

    def variant_cfg(self, variant: str) -> dict:
        try:
            v = dict(self.cfg["variants"][variant])
        except KeyError:
            raise SystemExit(f"unknown variant {variant!r}; config has {sorted(self.cfg['variants'])}")
        for k in self.PER_VARIANT:
            v.setdefault(k, self.cfg.get(k))
        v["served_name"] = v["served_name"] or self.served_name
        v["min_tokens"] = True if v["min_tokens"] is None else v["min_tokens"]
        return v

    def _env(self, variant: str) -> dict:
        v = self.variant_cfg(variant)
        env = dict(os.environ)
        env.update({
            "POC_VARIANT": variant,
            "POC_SERVED_NAME": v["served_name"],
            "POC_WEIGHTS": v["weights"],
            "POC_ARGS": json.dumps(v["args"]),
            "POC_GPU": str(v["gpu"]),
            "POC_CPU": str(v["cpu_cores"]),
            "POC_MEMORY_GIB": str(v["memory_gib"]),
            "POC_VLLM": str(v["vllm"] or "0.11.0"),
            "POC_TRANSFORMERS": str(v["transformers"] or ""),
            "POC_MAX_MODEL_LEN": str(v["max_model_len"] or 12288),
        })
        return env

    def deploy(self, variant: str) -> None:
        subprocess.run(["modal", "deploy", str(APP_FILE)], env=self._env(variant), check=True)

    def url(self, variant: str) -> str:
        import modal

        fn = modal.Function.from_name(app_name(variant), "serve")
        return fn.get_web_url().rstrip("/")

    def start(self, variant: str) -> Endpoint:
        key = os.environ.get("POC_VLLM_KEY")
        if not key:
            raise SystemExit("POC_VLLM_KEY is not set; scripts/setup_modal.sh writes it to .env.")
        self.deploy(variant)
        base = self.url(variant)
        return Endpoint(
            provider=self.name,
            variant=variant,
            base_url=f"{base}/v1",
            model=self.variant_cfg(variant)["served_name"],
            headers={"Authorization": f"Bearer {key}"},
            metrics_url=f"{base}/metrics",
            honours_min_tokens=bool(self.variant_cfg(variant)["min_tokens"]),
        )

    def stop(self, variant: str) -> None:
        """Stop the app, then check Modal no longer lists it. Loud on failure: a running app bills."""
        name = app_name(variant)
        r = subprocess.run(["modal", "app", "stop", "-y", name], capture_output=True, text=True)
        if r.returncode != 0 and "not found" not in (r.stderr + r.stdout).lower():
            print(f"!! modal app stop {name} failed: {(r.stderr or r.stdout).strip()}")
        deadline = time.time() + 90
        while True:
            still = [a for a in running_poc_apps() if a.startswith(name + " ")]
            if not still or time.time() > deadline:
                break
            time.sleep(5)
        if still:
            print(f"!! STILL RUNNING: {still}. Stop it now: modal app stop -y {name}")
        else:
            print(f"  stopped {name}")

    def usd_per_sec(self, variant: str | None = None) -> float:
        """Container price per second. Each variant carries its own GPU price: an H100 costs twice an L40S."""
        c = self.variant_cfg(variant) if variant else self.cfg
        return (c["usd_per_sec_gpu"]
                + c["cpu_cores"] * self.cfg["usd_per_sec_core"]
                + c["memory_gib"] * self.cfg["usd_per_sec_gib"])

    def run_cost_usd(self, rows: list[dict], billed_seconds: float, variant: str | None = None) -> float:
        # Container-seconds from the first request to the stop: an upper bound on what Modal bills.
        return billed_seconds * self.usd_per_sec(variant)

    def usd_per_1k_requests(self, req_per_s: float, utilization: float = 1.0, variant: str | None = None) -> float:
        """GPU cost per 1,000 requests at a measured request rate, if the GPU is busy `utilization` of the time."""
        if req_per_s <= 0 or utilization <= 0:
            return float("nan")
        return self.usd_per_sec(variant) / (req_per_s * utilization) * 1000
