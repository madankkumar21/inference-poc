"""Stop everything this repo can leave billing: Together dedicated endpoints and Modal apps.

    uv run python scripts/stop_all.py           # stop and delete, then list what is left
    uv run python scripts/stop_all.py --check   # list only; exit 1 if anything is running

run_poc.py runs the check before every paid phase and refuses to start while anything is left.
Run this by hand if a run was killed, the laptop slept, or a terminal closed mid-phase.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from providers.modal_vllm import running_poc_apps   # noqa: E402


def leftovers(cfg: dict) -> list[str]:
    """Names of anything still running: Modal apps not stopped, and Together endpoints this repo made."""
    out = [f"modal: {a}" for a in running_poc_apps()]
    if cfg["providers"].get("together_dedicated"):
        from providers.together_dedicated import TogetherDedicated
        out += [f"together: {e['name']}" for e in TogetherDedicated(cfg["providers"]["together_dedicated"]).poc_endpoints()]
    return out


def stop_everything(cfg: dict) -> list[str]:
    import subprocess
    for a in running_poc_apps():
        name = a.split(" ")[0]
        subprocess.run(["modal", "app", "stop", "-y", name], capture_output=True, text=True)
        print(f"  asked Modal to stop {name}")
    if cfg["providers"].get("together_dedicated"):
        from providers.together_dedicated import TogetherDedicated
        TogetherDedicated(cfg["providers"]["together_dedicated"]).stop()
    return leftovers(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load((ROOT / "config" / "poc.yaml").read_text())
    left = leftovers(cfg) if args.check else stop_everything(cfg)
    print("nothing running" if not left else "STILL RUNNING:\n  " + "\n  ".join(left))
    sys.exit(1 if left else 0)


if __name__ == "__main__":
    main()
