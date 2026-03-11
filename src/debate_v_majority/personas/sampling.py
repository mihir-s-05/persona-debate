from __future__ import annotations

import math
import random

from .schema import Axis


def _distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) | set(b))
    return math.sqrt(sum((float(a.get(k, 0.5)) - float(b.get(k, 0.5))) ** 2 for k in keys))


def _halton(index: int, base: int) -> float:
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def _halton_points(*, axis_ids: list[str], num_personas: int, seed: int) -> list[dict[str, float]]:
    # Offset the sequence by seed to keep deterministic but non-identical runs.
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    start = max(1, seed + 1)
    points: list[dict[str, float]] = []
    for offset in range(num_personas):
        idx = start + offset
        points.append(
            {
                axis_id: round(_halton(idx, primes[j % len(primes)]), 4)
                for j, axis_id in enumerate(axis_ids)
            }
        )
    return points


def _maximin_points(*, axis_ids: list[str], num_personas: int, seed: int) -> list[dict[str, float]]:
    rng = random.Random(seed)
    points: list[dict[str, float]] = []

    def random_point() -> dict[str, float]:
        return {axis_id: round(rng.random(), 4) for axis_id in axis_ids}

    candidates = [random_point() for _ in range(max(64, num_personas * 16))]
    points.append(candidates[0])
    while len(points) < num_personas:
        best = None
        best_score = -1.0
        for cand in candidates:
            score = min(_distance(cand, existing) for existing in points)
            if score > best_score:
                best = cand
                best_score = score
        assert best is not None
        points.append(best)
        candidates.remove(best)
        if not candidates:
            candidates = [random_point() for _ in range(max(32, num_personas * 8))]
    return points


def sample_axis_points(
    *,
    axes: list[Axis],
    num_personas: int,
    seed: int,
    method: str = "maximin",
) -> list[dict[str, float]]:
    if num_personas <= 0:
        return []
    axis_ids = [axis.axis_id for axis in axes]
    if not axis_ids:
        return [{} for _ in range(num_personas)]
    if method == "halton":
        return _halton_points(axis_ids=axis_ids, num_personas=num_personas, seed=seed)
    return _maximin_points(axis_ids=axis_ids, num_personas=num_personas, seed=seed)
