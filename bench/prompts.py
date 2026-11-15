"""Synthetic prompts at a target token count.

Word counts are scaled from a measured tokens-per-word ratio (calibrate() uses the first response's
usage.prompt_tokens), so prompts land near the target on any tokenizer. Every row records the real count.
"""

from __future__ import annotations

import random
import uuid

WORDS = (
    "account address agent alert archive audit backup balance batch bridge budget buffer cache "
    "capacity carrier channel chart check client cluster column commit compute config contract "
    "control copy cost counter credit cycle data deadline deal debit delay demand deploy device "
    "disk domain draft driver engine error estimate event export factor failover feature field "
    "filter fleet flow forecast format frame gateway graph handler header health history host "
    "image index input install invoice issue job journal kernel key label latency layer ledger "
    "limit link load lock log margin market memory merge message meter metric migration mirror "
    "model module monitor network node notice object offset order output owner packet page "
    "partner patch path payload peak pipeline plan policy pool port price process profile "
    "project prompt protocol proxy query queue quota range rate record region release replica "
    "report request reserve resource response result retry review risk route rule runtime "
    "sample scale schema score script search segment server service session shard signal "
    "snapshot source stage state status storage stream subnet supply switch system table target "
    "task tenant thread ticket tier timeout token topic trace traffic trigger update upgrade "
    "usage user value vendor version volume window worker workload zone"
).split()

DEFAULT_TOKENS_PER_WORD = 1.25


def _words(n: int, rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(max(n, 1)))


def seed_for(*parts) -> int:
    """A fixed seed from (workload, level, index), so every arm gets the same prompts in the same order."""
    import hashlib
    return int(hashlib.sha256(repr(parts).encode()).hexdigest()[:12], 16)


def quality_items(n: int, seed: int = 11) -> list[tuple[str, str]]:
    """Arithmetic with one right answer, to check a quantized or served model still answers the same."""
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        a, b, c = rng.randint(12, 99), rng.randint(12, 99), rng.randint(2, 9)
        items.append((f"What is {a} * {b} + {c}? Reply with the number only.", str(a * b + c)))
    return items


def chat_prompt(target_tokens: int, tokens_per_word: float, seed: int) -> str:
    """A unique prompt: the request id comes first, so no two prompts share a cached prefix."""
    rng = random.Random(seed)
    body = _words(int(target_tokens / tokens_per_word) - 40, rng)
    return (f"[request {uuid.UUID(int=rng.getrandbits(128))}] Read these operations notes, then write a long, "
            f"detailed incident report about them. Keep writing until you reach the length limit.\n\n{body}")


class SharedDocument:
    """One long document reused by every request; only the closing question differs."""

    def __init__(self, target_tokens: int, tokens_per_word: float, seed: int = 7):
        self.doc = _words(int(target_tokens / tokens_per_word) - 60, random.Random(seed))

    def prompt(self, i: int) -> str:
        topic = WORDS[(i * 37) % len(WORDS)]
        return (f"Operations log:\n\n{self.doc}\n\nQuestion {i}: write a long, detailed report on every "
                f"entry in the log above that mentions '{topic}'. Keep writing until you reach the length limit.")


def calibrate(sent_words: int, prompt_tokens: int, overhead_tokens: int = 60) -> float:
    """Tokens per word from one response; chat-template overhead is removed first."""
    if sent_words <= 0 or prompt_tokens <= overhead_tokens:
        return DEFAULT_TOKENS_PER_WORD
    return (prompt_tokens - overhead_tokens) / sent_words
