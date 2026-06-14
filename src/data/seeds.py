"""Deterministic RNG helpers.

The whole synthetic world derives from one integer seed. Each archetype / phase
draws its own independent generator via `spawn(seed, name)` so adding or
reordering one archetype does not perturb another's stream — important for
reproducible experiments.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _mix(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()
    return int(digest, 16) % (2**32)


def spawn(seed: int, name: str) -> np.random.Generator:
    """A named, independent generator deterministically derived from the seed."""
    return np.random.default_rng(_mix(seed, name))
