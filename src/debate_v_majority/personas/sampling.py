from __future__ import annotations

import math
import random

from .schema import Axis


GENERIC_SLOT_ROLES = (
    "parallel_explorer",
    "committed_builder",
    "global_theorist",
    "local_verifier",
    "skeptical_falsifier",
)

_BASE_PROFILE = {
    "low": 0.16,
    "high": 0.84,
    "mid_low": 0.34,
    "mid_high": 0.66,
    "center": 0.50,
    "jitter": 0.05,
}

_ROLE_TARGETS: dict[str, dict[str, float]] = {
    "parallel_explorer": {
        "commitment_style": _BASE_PROFILE["low"],
        "search_strategy": _BASE_PROFILE["mid_high"],
        "verification_timing": _BASE_PROFILE["mid_high"],
        "social_update_style": _BASE_PROFILE["low"],
    },
    "committed_builder": {
        "commitment_style": _BASE_PROFILE["high"],
        "search_strategy": _BASE_PROFILE["low"],
        "abstraction_preference": _BASE_PROFILE["mid_low"],
        "verification_timing": _BASE_PROFILE["mid_low"],
    },
    "global_theorist": {
        "abstraction_preference": _BASE_PROFILE["high"],
        "evidence_preference": _BASE_PROFILE["high"],
        "verification_timing": _BASE_PROFILE["mid_low"],
        # Keep the theorist in a balanced commitment band so the fixed slot bank
        # does not systematically violate the coverage audit's low/high caps.
        "commitment_style": _BASE_PROFILE["center"],
    },
    "local_verifier": {
        "abstraction_preference": _BASE_PROFILE["low"],
        "evidence_preference": _BASE_PROFILE["low"],
        "verification_timing": _BASE_PROFILE["high"],
        "commitment_style": _BASE_PROFILE["mid_high"],
    },
    "skeptical_falsifier": {
        "search_strategy": _BASE_PROFILE["high"],
        "verification_timing": _BASE_PROFILE["high"],
        "social_update_style": _BASE_PROFILE["mid_high"],
        # The falsifier should preserve skeptical pruning without becoming a
        # third low-commitment slot by construction.
        "commitment_style": _BASE_PROFILE["center"],
    },
}


def slot_role_for_index(slot_index: int) -> str:
    return GENERIC_SLOT_ROLES[slot_index % len(GENERIC_SLOT_ROLES)]


def _distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) | set(b))
    return math.sqrt(sum((float(a.get(k, 0.5)) - float(b.get(k, 0.5))) ** 2 for k in keys))


def _clamp(value: float) -> float:
    return min(0.98, max(0.02, value))


def _halton(index: int, base: int) -> float:
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def _axis_default(axis_id: str) -> float:
    if axis_id == "social_update_style":
        return _BASE_PROFILE["mid_low"]
    return _BASE_PROFILE["center"]


def _candidate_point(
    *,
    axis_ids: list[str],
    role: str,
    seed: int,
    slot_index: int,
    candidate_index: int,
    method: str,
) -> dict[str, float]:
    role_targets = _ROLE_TARGETS.get(role, {})
    point: dict[str, float] = {}
    for axis_index, axis_id in enumerate(axis_ids):
        base = role_targets.get(axis_id, _axis_default(axis_id))
        if method == "halton":
            prime = [2, 3, 5, 7, 11, 13, 17, 19][axis_index % 8]
            jitter = (_halton(seed + slot_index * 31 + candidate_index * 17 + axis_index + 1, prime) - 0.5) * _BASE_PROFILE["jitter"]
        else:
            rng = random.Random(seed + slot_index * 1009 + candidate_index * 7919 + axis_index * 97)
            jitter = (rng.random() - 0.5) * _BASE_PROFILE["jitter"]
        point[axis_id] = round(_clamp(base + jitter), 4)
    return point


def sample_axis_points(
    *,
    axes: list[Axis],
    num_personas: int,
    seed: int,
    method: str = "maximin",
    benchmark_family: str | None = None,
) -> list[dict[str, float]]:
    _ = benchmark_family
    if num_personas <= 0:
        return []
    axis_ids = [axis.axis_id for axis in axes]
    if not axis_ids:
        return [{} for _ in range(num_personas)]

    selected: list[dict[str, float]] = []
    candidate_count = 12
    for slot_index in range(num_personas):
        role = slot_role_for_index(slot_index)
        candidates = [
            _candidate_point(
                axis_ids=axis_ids,
                role=role,
                seed=seed,
                slot_index=slot_index,
                candidate_index=candidate_index,
                method=method,
            )
            for candidate_index in range(candidate_count)
        ]
        target = _candidate_point(
            axis_ids=axis_ids,
            role=role,
            seed=seed,
            slot_index=slot_index,
            candidate_index=0,
            method=method,
        )
        if method == "halton" or not selected:
            best = min(candidates, key=lambda cand: _distance(cand, target))
        else:
            best = max(
                candidates,
                key=lambda cand: min(_distance(cand, prev) for prev in selected)
                - (0.20 * _distance(cand, target)),
            )
        selected.append(best)
    return selected
