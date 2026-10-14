"""Deterministic pooled Recall@K for paired CARE retrieval features."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


KS = (1, 5, 10)
POOL_SIZE = 16
TIE_POLICY = "competition: rank = 1 + count(candidate similarity > positive similarity)"


def _as_string_ids(values: np.ndarray, expected: int) -> list[str]:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != expected:
        raise ValueError(f"ids must be a length-{expected} vector, got {array.shape}")
    raw_ids = array.tolist()
    for index, value in enumerate(raw_ids):
        if value is None or (
            isinstance(value, (float, np.floating)) and np.isnan(value)
        ):
            raise ValueError(f"ids[{index}] is null or NaN")
    ids = [str(value).strip() for value in raw_ids]
    if any(not value for value in ids):
        raise ValueError("ids must be non-empty; metrics will not fabricate identities")
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be unique")
    return ids


def _normalized_features(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name} must be a non-empty rank-2 matrix, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError(f"{name} contains zero-norm or invalid rows")
    return array / norms


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    """Load new caches safely and legacy object-array ids only as a fallback."""
    cache_path = Path(path).expanduser().resolve()
    with np.load(cache_path, allow_pickle=False) as data:
        required = {"text_features", "image_features", "ids"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Feature cache is missing keys: {missing}")
        text = np.asarray(data["text_features"])
        image = np.asarray(data["image_features"])
        try:
            raw_ids = np.asarray(data["ids"])
            used_legacy_pickle = False
        except ValueError as exc:
            if "Object arrays cannot be loaded" not in str(exc):
                raise
            raw_ids = None
            used_legacy_pickle = True
        metadata_json = str(data["metadata_json"].item()) if "metadata_json" in data.files else None
        patient_keys = (
            [str(value) for value in np.asarray(data["patient_keys"]).tolist()]
            if "patient_keys" in data.files
            else None
        )

    if used_legacy_pickle:
        # Historical three-key CARE caches stored only ids as an object array.
        # Use this fallback only for trusted local caches.
        with np.load(cache_path, allow_pickle=True) as legacy:
            raw_ids = np.asarray(legacy["ids"])

    text = _normalized_features(text, "text_features")
    image = _normalized_features(image, "image_features")
    if text.shape != image.shape:
        raise ValueError(f"Text/image feature shapes differ: {text.shape} vs {image.shape}")
    ids = _as_string_ids(raw_ids, text.shape[0])
    if patient_keys is not None and len(patient_keys) != len(ids):
        raise ValueError("patient_keys length does not match features")

    metadata = None
    if metadata_json is not None:
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            raise ValueError("metadata_json is not valid JSON") from exc
    return {
        "text_features": text,
        "image_features": image,
        "ids": ids,
        "patient_keys": patient_keys,
        "metadata": metadata,
        "legacy_pickle_ids": used_legacy_pickle,
        "path": cache_path,
    }


def _partitions(n: int, count: int, seed: int) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("pool_partitions must be at least 1")
    complete = (n // POOL_SIZE) * POOL_SIZE
    if complete == 0:
        raise ValueError(f"At least {POOL_SIZE} paired candidates are required")
    output = []
    for partition in range(count):
        permutation = torch.randperm(
            n, generator=torch.Generator().manual_seed(seed + partition)
        ).numpy()
        output.append({
            "partition": partition,
            "pools": permutation[:complete].reshape(-1, POOL_SIZE),
            "dropped": permutation[complete:],
        })
    return output


def _candidate_ranks(scores: np.ndarray) -> np.ndarray:
    return np.asarray([1 + int(np.count_nonzero(scores > value)) for value in scores])


def _evaluate_direction(
    similarities: np.ndarray,
    ids: list[str],
    partitions: list[dict[str, Any]],
    csv_path: Path,
) -> dict[str, float]:
    positive_ranks: list[int] = []
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "partition", "pool_id", "query_id", "target_id", "rank",
            "similarity", "is_positive",
        ])
        writer.writeheader()
        for item in partitions:
            partition = int(item["partition"])
            for pool_id, pool in enumerate(item["pools"]):
                subset = similarities[np.ix_(pool, pool)]
                for local_query, query_index in enumerate(pool):
                    scores = subset[local_query]
                    ranks = _candidate_ranks(scores)
                    positive_ranks.append(int(ranks[local_query]))
                    order = np.argsort(-scores, kind="stable")
                    for local_target in order:
                        target_index = int(pool[local_target])
                        writer.writerow({
                            "partition": partition,
                            "pool_id": pool_id,
                            "query_id": ids[int(query_index)],
                            "target_id": ids[target_index],
                            "rank": int(ranks[local_target]),
                            "similarity": f"{float(scores[local_target]):.10g}",
                            "is_positive": int(local_target == local_query),
                        })
    ranks = np.asarray(positive_ranks)
    return {f"R@{k}": float(np.mean(ranks <= k) * 100.0) for k in KS}


def compute_pooled_metrics(
    text_features: np.ndarray,
    image_features: np.ndarray,
    ids: list[str],
    output_dir: str | Path,
    pool_partitions: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute R2V and V2R from the same deterministic disjoint partitions."""
    text = _normalized_features(text_features, "text_features")
    image = _normalized_features(image_features, "image_features")
    if text.shape != image.shape or len(ids) != text.shape[0]:
        raise ValueError("Paired text, image, and id counts must match")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    partitions = _partitions(len(ids), pool_partitions, seed)
    r2v = text @ image.T
    results = {
        "R2V": _evaluate_direction(r2v, ids, partitions, destination / "rankings_r2v.csv"),
        "V2R": _evaluate_direction(r2v.T, ids, partitions, destination / "rankings_v2r.csv"),
        "pool_size": POOL_SIZE,
        "pool_partitions": pool_partitions,
        "seed": seed,
        "n_pools_per_partition": len(partitions[0]["pools"]),
        "n_dropped": len(partitions[0]["dropped"]),
        "dropped_ids": [
            {
                "partition": int(item["partition"]),
                "ids": [ids[int(index)] for index in item["dropped"]],
            }
            for item in partitions
        ],
        "tie_policy": TIE_POLICY,
    }
    return results
