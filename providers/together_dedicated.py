"""Together dedicated endpoint: one replica on one GPU, billed per minute while it runs.

Lifecycle, through Together's v2 API:
    create endpoint -> create deployment (min = max = 1 replica) -> route traffic to it -> wait for READY
    ... benchmark ...
    scale to 0/0 -> wait for STOPPED -> delete deployment -> delete endpoint -> confirm nothing is left

A running deployment bills until it is stopped, so stop() is called from a finally block, the
deployment carries Together's own idle stops (scale-to-zero window and inactive timeout) in case
this process dies, and scripts/stop_all.py removes anything this repo left behind.
"""

from __future__ import annotations

import os
import time

import httpx

from providers.base import Endpoint, Provider

API = "https://api.together.ai"
PREFIX = "inference-poc-"   # every endpoint this repo creates starts with this; stop_all looks for it


class DedicatedError(RuntimeError):
    pass


class TogetherDedicated(Provider):
    name = "together_dedicated"
    self_hosted = False

    def __init__(self, cfg: dict, client: httpx.Client | None = None, poll_s: float = 10.0):
        self.cfg = cfg
        self.http = client or httpx.Client(base_url=API, timeout=60)
        self.poll_s = poll_s
        self._project: dict | None = None
        self.started_at: dict[str, float] = {}   # variant -> time the deployment was requested
        self.ready_at: dict[str, float] = {}     # variant -> time it reached READY

    # ---------- plumbing ----------
    def _headers(self) -> dict:
        key = os.environ.get("TOGETHER_API_KEY")
        if not key:
            raise SystemExit("TOGETHER_API_KEY is not set. Put it in .env (see .env.example).")
        return {"Authorization": f"Bearer {key}"}

    def _req(self, method: str, path: str, **kw) -> dict:
        r = self.http.request(method, path, headers=self._headers(), **kw)
        if r.status_code >= 400:
            raise DedicatedError(f"{method} {path}: HTTP {r.status_code}: {r.text[:400]}")
        return r.json() if r.content else {}

    def project(self) -> dict:
        if self._project is None:
            who = self._req("GET", "/v3/whoami")
            if who.get("type") != "project_api_key" or not who.get("project"):
                raise SystemExit("TOGETHER_API_KEY must be a project API key for dedicated endpoints")
            self._project = who["project"]
        return self._project

    def _p(self, path: str = "") -> str:
        return f"/v2/projects/{self.project()['id']}{path}"

    def variant_cfg(self, variant: str) -> dict:
        try:
            return self.cfg["variants"][variant]
        except KeyError:
            raise SystemExit(f"unknown dedicated variant {variant!r}; config has {sorted(self.cfg['variants'])}")

    # ---------- what exists ----------
    def poc_endpoints(self) -> list[dict]:
        """Endpoints this repo created (name starts with PREFIX), with their deployments."""
        data = self._req("GET", self._p("/endpoints")).get("data") or []
        return [e for e in data if e.get("name", "").split("/")[-1].startswith(PREFIX)]

    def deployments(self, endpoint_id: str) -> list[dict]:
        return self._req("GET", self._p(f"/endpoints/{endpoint_id}/deployments")).get("data") or []

    # ---------- start ----------
    def start(self, variant: str) -> Endpoint:
        v = self.variant_cfg(variant)
        left = self.poc_endpoints()
        if left:
            raise DedicatedError(f"refusing to start: endpoints from an earlier run still exist: "
                                 f"{[e['name'] for e in left]}. Run scripts/stop_all.py first.")
        ep = self._req("POST", self._p("/endpoints"), json={"name": PREFIX + variant})
        self.started_at[variant] = time.time()
        try:
            dep = self._req("POST", self._p(f"/endpoints/{ep['id']}/deployments"), json={
                "name": variant,
                "model": v["model"],
                "config": v["config"],
                "autoscaling": {"minReplicas": 1, "maxReplicas": 1,
                                "scaleToZeroWindow": f"{int(self.cfg['scale_to_zero_s'])}s"},
                "inactiveTimeout": int(self.cfg["inactive_timeout_min"]),
            })
            self._req("PATCH", self._p(f"/endpoints/{ep['id']}"), params={"updateMask": "trafficSplit"},
                      json={"trafficSplit": [{"deploymentId": dep["id"], "weight": 1}]})
            self._wait(ep["id"], dep["id"], "DEPLOYMENT_STATE_READY", self.cfg["ready_timeout_s"])
            self.ready_at[variant] = time.time()
        except BaseException:
            self.stop(variant)
            raise
        return Endpoint(
            provider=self.name,
            variant=variant,
            base_url=self.cfg["base_url"],
            model=ep["name"],                       # "<project_slug>/<endpoint_name>" is the model string
            headers=self._headers(),
            extra_body=dict(self.cfg.get("extra_body") or {}),
        )

    def _wait(self, endpoint_id: str, dep_id: str, want: str, timeout_s: float) -> dict:
        deadline = time.time() + timeout_s
        while True:
            d = self._req("GET", self._p(f"/endpoints/{endpoint_id}/deployments/{dep_id}"))
            state = (d.get("status") or {}).get("state", "")
            if state == want:
                return d
            if state == "DEPLOYMENT_STATE_FAILED":
                raise DedicatedError(f"deployment {dep_id} failed: {(d.get('status') or {}).get('message')}")
            if time.time() > deadline:
                raise DedicatedError(f"deployment {dep_id} not {want} after {timeout_s:.0f} s (last state {state})")
            time.sleep(self.poll_s)

    # ---------- stop ----------
    def stop(self, variant: str | None = None) -> list[str]:
        """Stop and delete every endpoint this repo created (or only `variant`'s). Returns what is left."""
        for e in self.poc_endpoints():
            if variant and e["name"].split("/")[-1] != PREFIX + variant:
                continue
            for d in self.deployments(e["id"]):
                try:
                    self._req("PATCH", self._p(f"/endpoints/{e['id']}/deployments/{d['id']}"),
                              params={"updateMask": "autoscaling.minReplicas,autoscaling.maxReplicas"},
                              json={"autoscaling": {"minReplicas": 0, "maxReplicas": 0}})
                except DedicatedError as err:
                    print(f"!! scale-to-0 failed for {d['id']}: {err}")
            try:
                self._req("PATCH", self._p(f"/endpoints/{e['id']}"), params={"updateMask": "trafficSplit"},
                          json={"trafficSplit": []})
            except DedicatedError as err:
                print(f"!! clearing traffic failed for {e['id']}: {err}")
            for d in self.deployments(e["id"]):
                try:
                    self._wait(e["id"], d["id"], "DEPLOYMENT_STATE_STOPPED", self.cfg["stop_timeout_s"])
                    self._req("DELETE", self._p(f"/endpoints/{e['id']}/deployments/{d['id']}"))
                except DedicatedError as err:
                    print(f"!! stopping/deleting deployment {d['id']} failed: {err}")
            try:
                self._req("DELETE", self._p(f"/endpoints/{e['id']}"))
            except DedicatedError as err:
                print(f"!! deleting endpoint {e['id']} failed: {err}")
        left = [e["name"] for e in self.poc_endpoints()
                if not variant or e["name"].split("/")[-1] == PREFIX + variant]
        if left:
            print(f"!! STILL PRESENT on Together: {left}. Check the console; run scripts/stop_all.py.")
        else:
            print(f"  stopped and deleted Together dedicated {variant or 'endpoints'}")
        return left

    # ---------- cost ----------
    def usd_per_sec(self, variant: str | None = None) -> float:
        v = self.variant_cfg(variant) if variant else {}
        return self.cfg["usd_per_gpu_hour"] * v.get("gpu_count", 1) / 3600

    def run_cost_usd(self, rows: list[dict], billed_seconds: float, variant: str | None = None) -> float:
        # Seconds from the deployment request to the stop: an upper bound, since startup may not bill.
        return billed_seconds * self.usd_per_sec(variant)

    def usd_per_1k_requests(self, req_per_s: float, utilization: float = 1.0, variant: str | None = None) -> float:
        if req_per_s <= 0 or utilization <= 0:
            return float("nan")
        return self.usd_per_sec(variant) / (req_per_s * utilization) * 1000
