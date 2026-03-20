from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .. import DatasetName
from ..datasets import get_dataset_adapter as resolve_dataset_adapter, get_registry_entry


def _parse_csv_ints(s: str | None) -> list[int]:
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _parse_subset_n_arg(v: str) -> int | str:
    """
    Parse --subset_n which normally is an int, but also supports "all"/"*"
    to mean "use the full dataset" (equivalent to --all).
    """
    s = str(v).strip().lower()
    if s in ("all", "*"):
        return "all"
    return int(s)


@dataclass(frozen=True)
class SubsetItem:
    """A single item in the evaluation subset."""

    subset_id: int
    orig_id: int
    item_uid: str
    dataset_revision: str | None
    item_display_id: str | int | None
    raw_task: dict[str, Any]
    dataset_meta: dict[str, Any]
    family: str | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dataset_test_path_candidates(
    dataset: DatasetName,
    *,
    source_file: Path | None = None,
    hle_variant: str | None = None,
    dataset_local_mirror: Path | None = None,
) -> list[Path]:
    source = source_file or Path(__file__).resolve()
    repo_root = source.parents[3]
    package_root = source.parents[1]
    legacy_cli_root = source.parent
    registry_entry = get_registry_entry(dataset)
    variant = str(hle_variant or "verified").strip().lower()
    filename = (
        registry_entry.variant_filenames.get(variant, registry_entry.default_filename)
        if registry_entry.variant_filenames
        else registry_entry.default_filename
    )
    if dataset_local_mirror is not None:
        if dataset_local_mirror.suffix.lower() == ".jsonl":
            return [dataset_local_mirror]
        return [dataset_local_mirror / dataset / filename]
    return [
        repo_root / "data" / dataset / filename,
        package_root / "data" / dataset / filename,
        legacy_cli_root / "data" / dataset / filename,
    ]


def _default_dataset_test_path(
    dataset: DatasetName,
    *,
    hle_variant: str | None = None,
    dataset_local_mirror: Path | None = None,
) -> Path:
    candidates = _dataset_test_path_candidates(
        dataset,
        hle_variant=hle_variant,
        dataset_local_mirror=dataset_local_mirror,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_exclude_id_map(path: Path | None) -> dict[str, str | None]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {
            line.strip(): None
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
    if isinstance(payload, list):
        return {str(item).strip(): None for item in payload if str(item).strip()}
    if isinstance(payload, dict):
        return {
            str(key).strip(): (None if value is None else str(value))
            for key, value in payload.items()
            if str(key).strip()
        }
    raise ValueError(f"Unsupported exclude-id payload in {path}")


def _canonical_item_identifiers(task_item: "SubsetItem | Any") -> list[str]:
    raw_task = getattr(task_item, "raw_task", {}) or {}
    display_id = getattr(task_item, "item_display_id", None)
    if display_id is None:
        display_id = getattr(task_item, "display_id", None)
    candidates = [
        str(getattr(task_item, "item_uid", "")).strip(),
        "" if display_id is None else str(display_id).strip(),
        "" if raw_task.get("id") is None else str(raw_task.get("id")).strip(),
        "" if raw_task.get("item_uid") is None else str(raw_task.get("item_uid")).strip(),
    ]
    seen: set[str] = set()
    out: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def _ensure_dataset_test_jsonl(
    dataset: DatasetName,
    test_path: Path,
    *,
    hle_variant: str = "verified",
) -> None:
    """Download dataset if not present."""
    if test_path.exists():
        return

    print(f"[data] Downloading {dataset} from HuggingFace -> {test_path}", file=sys.stderr)
    adapter = resolve_dataset_adapter(dataset)
    adapter.materialize(test_path, variant=str(hle_variant))


def _matches_hle_modality(task_item: Any, *, hle_modality: str) -> bool:
    mode = str(hle_modality or "any").strip().lower()
    if mode == "any":
        return True
    from ..datasets import hle as hle_dataset

    has_images = bool(hle_dataset.has_images(getattr(task_item, "raw_task", {}) or {}))
    if mode == "text_only":
        return not has_images
    if mode == "image_only":
        return has_images
    raise ValueError(f"Unsupported HLE modality filter: {hle_modality}")


def _make_dataset_subset(
    *,
    dataset: DatasetName,
    test_path: Path,
    n: int,
    seed: int,
    ids: list[int] | None,
    range_str: str | None,
    hle_variant: str = "verified",
    hle_modality: str = "any",
    exclude_id_map: dict[str, str | None] | None = None,
    exclude_ids_path: Path | None = None,
) -> tuple[list[SubsetItem], dict[str, Any]]:
    """Create a subset of the dataset."""
    exclude_id_map = dict(exclude_id_map or {})
    _ensure_dataset_test_jsonl(dataset, test_path, hle_variant=hle_variant)
    adapter = resolve_dataset_adapter(dataset)
    registry_entry = get_registry_entry(dataset)
    load_result = adapter.load_items(
        test_path,
        registry_meta={
            "source_dataset_id": registry_entry.source_dataset_id,
            "source_dataset_config": registry_entry.source_dataset_config,
            "source_dataset_split": registry_entry.source_dataset_split,
            "source_dataset_revision": registry_entry.source_dataset_revision,
        },
    )
    all_items = load_result.items
    total_before_modality = len(all_items)
    if dataset == "hle":
        all_items = [
            task_item
            for task_item in all_items
            if _matches_hle_modality(task_item, hle_modality=hle_modality)
        ]
    total = len(all_items)
    dataset_revision = load_result.dataset_revision
    canonical_identifiers = {
        candidate
        for task_item in all_items
        for candidate in _canonical_item_identifiers(task_item)
    }

    def _exclude_hit(orig_id: int, task_item) -> tuple[bool, str | None]:
        for candidate in _canonical_item_identifiers(task_item):
            if candidate and candidate in exclude_id_map:
                return True, exclude_id_map[candidate]
        orig_candidate = str(orig_id)
        if orig_candidate not in canonical_identifiers and orig_candidate in exclude_id_map:
            return True, exclude_id_map[orig_candidate]
        return False, None

    eligible_orig_ids = [
        orig_id
        for orig_id, task_item in enumerate(all_items)
        if not _exclude_hit(orig_id, task_item)[0]
    ]
    eligible_id_set = set(eligible_orig_ids)

    if ids:
        chosen_pairs: list[tuple[int, int]] = []
        for raw_idx in list(dict.fromkeys(ids)):
            resolved_idx = raw_idx if raw_idx >= 0 else total + raw_idx
            if resolved_idx < 0 or resolved_idx >= total:
                raise IndexError(f"subset id {raw_idx} out of range for dataset of size {total}")
            if resolved_idx in eligible_id_set:
                chosen_pairs.append((raw_idx, resolved_idx))
    elif range_str:
        s = str(range_str).strip().lower()
        if s in ("all", "*"):
            chosen_pairs = [(orig_id, orig_id) for orig_id in eligible_orig_ids]
        elif ":" in range_str:
            a, b = range_str.split(":", 1)
            start = int(a) if a else 0
            end = int(b) if b else total
            chosen_pairs = [
                (orig_id, orig_id)
                for orig_id in range(start, min(end, total))
                if orig_id in eligible_id_set
            ]
        else:
            a, b = range_str.split("-", 1)
            start = int(a)
            end = int(b) + 1
            chosen_pairs = [
                (orig_id, orig_id)
                for orig_id in range(start, min(end, total))
                if orig_id in eligible_id_set
            ]
    else:
        n = min(n, len(eligible_orig_ids))
        rng = random.Random(seed)
        chosen_pairs = [(orig_id, orig_id) for orig_id in rng.sample(eligible_orig_ids, n)]

    excluded_ids = [
        str(task_item.display_id) if task_item.display_id is not None else str(orig_id)
        for orig_id, task_item in enumerate(all_items)
        if _exclude_hit(orig_id, task_item)[0]
    ]
    meta = {
        "dataset": dataset,
        "dataset_revision": dataset_revision,
        "dataset_variant": hle_variant if dataset == "hle" else None,
        "hle_modality": hle_modality if dataset == "hle" else None,
        "dataset_config": registry_entry.source_dataset_config,
        "source_dataset_id": registry_entry.source_dataset_id,
        "source_dataset_split": registry_entry.source_dataset_split,
        "source_dataset_revision": registry_entry.source_dataset_revision,
        "total_available": total,
        "total_available_before_modality": total_before_modality if dataset == "hle" else total,
        "available_after_exclusion": len(eligible_orig_ids),
        "subset_size": 0,
        "seed": seed,
        "orig_ids": [orig_id for orig_id, _resolved_idx in chosen_pairs],
        "source_path": str(test_path),
        "exclude_ids_path": str(exclude_ids_path) if exclude_ids_path is not None else None,
        "excluded_count": len(excluded_ids),
        "excluded_ids": excluded_ids,
    }

    items: list[SubsetItem] = []
    for orig_id, resolved_idx in chosen_pairs:
        task_item = all_items[resolved_idx]
        raw_task = task_item.raw_task
        excluded, _reason = _exclude_hit(resolved_idx, task_item)
        if excluded:
            continue
        items.append(
            SubsetItem(
                subset_id=len(items),
                orig_id=orig_id,
                item_uid=task_item.item_uid,
                dataset_revision=task_item.dataset_revision,
                item_display_id=task_item.display_id,
                family=task_item.family,
                raw_task=raw_task,
                dataset_meta=dict(meta),
            )
        )
    meta["subset_size"] = len(items)
    for item in items:
        item.dataset_meta.update(meta)
    return items, meta


def _select_one_item(items: list[SubsetItem], wanted: str) -> list[SubsetItem]:
    target = str(wanted)
    canonical_matches = [item for item in items if target in _canonical_item_identifiers(item)]
    if canonical_matches:
        return [canonical_matches[0]]

    orig_matches = [item for item in items if target == str(item.orig_id)]
    if orig_matches:
        return [orig_matches[0]]

    subset_matches = [item for item in items if target == str(item.subset_id)]
    if subset_matches:
        return [subset_matches[0]]

    raise ValueError(f"--one did not match any item: {wanted}")


__all__ = [
    "SubsetItem",
    "_canonical_item_identifiers",
    "_default_dataset_test_path",
    "_ensure_dataset_test_jsonl",
    "_load_exclude_id_map",
    "_make_dataset_subset",
    "_parse_csv_ints",
    "_parse_subset_n_arg",
    "_read_jsonl",
    "_select_one_item",
    "_write_jsonl",
]
