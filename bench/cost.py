"""Running spend for the whole repo, with a hard budget stop.

One ledger file (results/spend.jsonl) covers every run id, so a new run cannot reset the budget.
"""

from __future__ import annotations

import json
from pathlib import Path


class OverBudget(RuntimeError):
    pass


class Ledger:
    def __init__(self, budget_usd: float, path: Path, run_id: str = ""):
        self.budget = budget_usd
        self.path = path
        self.run_id = run_id
        path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []
        if path.exists():
            self.entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    @property
    def spent(self) -> float:
        return sum(e["usd"] for e in self.entries)

    def for_run(self, run_id: str) -> list[dict]:
        return [e for e in self.entries if e.get("run_id") == run_id]

    def add(self, phase: str, usd: float, note: str = "") -> None:
        e = {"run_id": self.run_id, "phase": phase, "usd": round(usd, 4), "note": note}
        self.entries.append(e)
        with self.path.open("a") as f:
            f.write(json.dumps(e) + "\n")
        print(f"  spend: {phase} ${usd:.3f} ({note}); repo total ${self.spent:.2f} of ${self.budget:.2f}")

    def check(self, phase: str, expected_usd: float = 0.0) -> None:
        """Before a phase: refuse to start if the expected spend would pass the budget."""
        if self.spent + expected_usd > self.budget:
            raise OverBudget(f"{phase}: spent ${self.spent:.2f} + expected ${expected_usd:.2f} "
                             f"passes the ${self.budget:.2f} budget; not starting it")

    def check_running(self, phase: str, phase_so_far_usd: float) -> None:
        """During a phase: stop at the next level boundary once the running total passes the budget."""
        if self.spent + phase_so_far_usd > self.budget:
            raise OverBudget(f"{phase}: spent ${self.spent:.2f} + this phase ${phase_so_far_usd:.2f} "
                             f"passes the ${self.budget:.2f} budget; stopping")
