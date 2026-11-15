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

    def variant_cfg(self, variant: str) -> dict:
        try:
            return self.cfg["variants"][variant]
        except KeyError:
            raise SystemExit(f"unknown variant {variant!r}; config has {sorted(self.cfg['variants'])}")

    def _env(self, variant: str) -> dict:
        v = self.variant_cfg(variant)
        env = dict(os.environ)
        env.update({
            "POC_VARIANT": variant,
            "POC_SERVED_NAME": self.served_name,
            "POC_WEIGHTS": v["weights"],
            "POC_ARGS": json.dumps(v["args"]),
            "POC_GPU": self.cfg["gpu"],
            "POC_CPU": str(self.cfg["cpu_cores"]),
            "POC_MEMORY_GIB": str(self.cfg["memory_gib"]),
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
            model=self.served_name,
            headers={"Authorization": f"Bearer {key}"},
            metrics_url=f"{base}/metrics",
            honours_min_tokens=True,
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

    def usd_per_sec(self) -> float:
        c = self.cfg
        return (c["usd_per_sec_gpu"]
                + c["cpu_cores"] * c["usd_per_sec_core"]
                + c["memory_gib"] * c["usd_per_sec_gib"])

    def run_cost_usd(self, rows: list[dict], billed_seconds: float) -> float:
        # Container-seconds from the first request to the stop: an upper bound on what Modal bills.
        return billed_seconds * self.usd_per_sec()

    def usd_per_1k_requests(self, req_per_s: float, utilization: float = 1.0) -> float:
        """GPU cost per 1,000 requests at a measured request rate, if the GPU is busy `utilization` of the time."""
        if req_per_s <= 0 or utilization <= 0:
            return float("nan")
        return self.usd_per_sec() / (req_per_s * utilization) * 1000
