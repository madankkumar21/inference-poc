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


# Natural-language requests, for speculative decoding: a speculator guesses the next tokens, and it
# guesses ordinary prose far better than the random-word reports above. Short input, 250 tokens out.
NATURAL_TOPICS = (
    "how DNS resolves a domain name", "how a TCP connection is opened and closed", "how a hash table handles collisions",
    "how HTTPS keeps a connection private", "how a database index speeds up a query", "how garbage collection works",
    "how a load balancer chooses a server", "how a CPU cache reduces memory latency", "how public-key encryption works",
    "how a compiler turns source code into machine code", "how version control tracks changes", "how a refrigerator keeps food cold",
    "how vaccines train the immune system", "how a bicycle stays upright", "how compound interest grows savings",
    "how a jet engine produces thrust", "how photosynthesis stores energy", "how a credit score is calculated",
    "how GPS finds your position", "how noise-cancelling headphones work", "how a bill becomes a law in the United States",
    "how a heat pump warms a house", "how bread rises", "how an electric car battery is charged",
    "how a search engine ranks pages", "how a transformer language model predicts the next word", "how tides happen",
    "how a mortgage is paid down over time", "how a camera sensor captures light", "how a recommendation system picks items",
    "how a blood test measures cholesterol", "how a solar panel makes electricity", "how a container differs from a virtual machine",
    "how weather forecasts are made", "how a password is stored safely", "how an elevator knows where to stop",
    "how a supply chain moves goods from a factory to a store", "how a Wi-Fi router shares one connection", "how coffee is decaffeinated",
    "how a spreadsheet recalculates formulas",
)
NATURAL_STYLES = (
    "Explain {t} to a curious high-school student. Use plain English, one everyday example, and short paragraphs.",
    "Write a clear, friendly explanation of {t} for a new employee. Cover the main steps in order.",
    "Describe {t}. Start with the big picture, then walk through what happens step by step, then give one common misconception.",
)


def natural_prompt(i: int) -> str:
    """Request i: one of 40 topics in one of 3 styles (120 distinct prompts, then they repeat)."""
    t = NATURAL_TOPICS[i % len(NATURAL_TOPICS)]
    s = NATURAL_STYLES[(i // len(NATURAL_TOPICS)) % len(NATURAL_STYLES)]
    return s.format(t=t)
