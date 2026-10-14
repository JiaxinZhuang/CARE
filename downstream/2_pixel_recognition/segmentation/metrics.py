from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import torch


KIPA22_LABELS = {
    0: "background",
    1: "renal_vein",
    2: "kidney",
    3: "renal_artery",
    4: "tumor",
}
KIPA22_GROUPS = {
    "focus_DSC": (0, 2, 4),
    "foreground_macro_DSC": (1, 2, 3, 4),
    "anatomy_group_DSC": (0, 1, 2, 3),
    "tumor_group_DSC": (0, 4),
}


def dice_per_class(prediction, target, valid_mask, num_classes=5):
    scores = np.full(num_classes, np.nan, dtype=np.float64)
    for class_id in range(num_classes):
        pred_mask = (prediction == class_id) & valid_mask
        target_mask = (target == class_id) & valid_mask
        denominator = pred_mask.sum() + target_mask.sum()
        if denominator:
            scores[class_id] = 2.0 * np.logical_and(pred_mask, target_mask).sum() / denominator
    return scores


def grouped_point_estimate(case_class_scores, class_ids):
    per_class = np.nanmean(case_class_scores[:, class_ids], axis=0)
    return float(np.nanmean(per_class))


def _class_map_batch(values: torch.Tensor, name: str) -> torch.Tensor:
    if values.ndim == 5 and values.shape[1] == 1:
        return values[:, 0]
    if values.ndim != 4:
        raise ValueError(
            f"Expected {name} [B,D,H,W] or [B,1,D,H,W], got {tuple(values.shape)}"
        )
    return values


def prepare_metric_batch(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim != 5:
        raise ValueError(f"Expected logits [B,C,D,H,W], got {tuple(logits.shape)}")
    if logits.shape[1] < 2:
        raise ValueError("Metric logits must contain at least two classes")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite logits cannot be scored")

    targets = _class_map_batch(labels, "labels").long()
    predictions = logits.argmax(dim=1)
    if targets.shape != predictions.shape:
        raise ValueError(
            f"Prediction/label shape mismatch: {tuple(predictions.shape)} != "
            f"{tuple(targets.shape)}"
        )
    if valid_mask is None:
        valid = torch.ones_like(targets, dtype=torch.bool)
    else:
        valid = _class_map_batch(valid_mask, "valid_mask").bool()
        if valid.shape != targets.shape:
            raise ValueError(
                f"valid_mask shape mismatch: {tuple(valid.shape)} != "
                f"{tuple(targets.shape)}"
            )
    return predictions, targets, valid


def _prepare_prediction_batch(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predictions = _class_map_batch(predictions, "predictions").long()
    targets = _class_map_batch(labels, "labels").long()
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction/label shape mismatch: {tuple(predictions.shape)} != "
            f"{tuple(targets.shape)}"
        )
    if valid_mask is None:
        valid = torch.ones_like(targets, dtype=torch.bool)
    else:
        valid = _class_map_batch(valid_mask, "valid_mask").bool()
        if valid.shape != targets.shape:
            raise ValueError(
                f"valid_mask shape mismatch: {tuple(valid.shape)} != "
                f"{tuple(targets.shape)}"
            )
    return predictions, targets, valid


def _normalize_case_ids(case_ids: Sequence[str] | str | None, batch_size: int):
    if case_ids is None:
        return [None] * batch_size
    if isinstance(case_ids, str):
        case_ids = [case_ids]
    case_ids = list(case_ids)
    if len(case_ids) != batch_size:
        raise ValueError(
            f"Expected {batch_size} case IDs, received {len(case_ids)}"
        )
    return [str(case_id) for case_id in case_ids]


def _nan_group(scores: np.ndarray, class_ids: Sequence[int]) -> float:
    selected = scores[list(class_ids)]
    if np.isnan(selected).all():
        return float("nan")
    return float(np.nanmean(selected))


class PolyMetrics:
    """Case-aware DSC accumulator; ``mode`` is retained for API compatibility."""

    def __init__(self, args=None, mode: str = "fast"):
        if mode not in {"fast", "full"}:
            raise ValueError(f"Unknown metrics mode: {mode}")
        dataset_name = str(getattr(args, "dataset_name", "kipa22")).lower()
        if dataset_name != "kipa22":
            raise ValueError("This release metrics slice supports only KiPA22")
        self.args = args
        self.mode = mode
        self.num_classes = int(getattr(args, "out_channels", 5))
        if self.num_classes != len(KIPA22_LABELS):
            raise ValueError("KiPA22 metrics require out_channels=5")
        self.reset()

    def reset(self):
        self.records: list[dict] = []

    def per_case(
        self,
        prediction,
        target,
        valid_mask=None,
        case_id: str | None = None,
    ) -> dict:
        prediction = np.asarray(prediction)
        target = np.asarray(target)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target shape mismatch: {prediction.shape} != {target.shape}"
            )
        if valid_mask is None:
            valid_mask = np.ones(target.shape, dtype=bool)
        else:
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if valid_mask.shape != target.shape:
                raise ValueError(
                    f"valid_mask shape mismatch: {valid_mask.shape} != {target.shape}"
                )
        valid_labels = target[valid_mask]
        valid_predictions = prediction[valid_mask]
        for name, values in (("target", valid_labels), ("prediction", valid_predictions)):
            if values.size and (values.min() < 0 or values.max() >= self.num_classes):
                raise ValueError(f"{name} contains values outside [0, {self.num_classes - 1}]")

        scores = dice_per_class(
            prediction, target, valid_mask, num_classes=self.num_classes
        )
        record = {
            "case_id": case_id,
            "DSC_per_class": {
                KIPA22_LABELS[class_id]: float(scores[class_id])
                for class_id in range(self.num_classes)
            },
        }
        for name, class_ids in KIPA22_GROUPS.items():
            record[name] = _nan_group(scores, class_ids)
        record["DSC"] = record["foreground_macro_DSC"]
        return record

    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        case_ids: Sequence[str] | str | None = None,
    ) -> list[dict]:
        predictions, targets, valid = _prepare_prediction_batch(
            predictions, labels, valid_mask
        )
        ids = _normalize_case_ids(case_ids, predictions.shape[0])
        new_records = []
        for batch_index, case_id in enumerate(ids):
            record = self.per_case(
                predictions[batch_index].detach().cpu().numpy(),
                targets[batch_index].detach().cpu().numpy(),
                valid[batch_index].detach().cpu().numpy(),
                case_id=case_id,
            )
            self.records.append(record)
            new_records.append(record)
        return new_records

    def update_from_logits(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        case_ids: Sequence[str] | str | None = None,
    ) -> list[dict]:
        predictions, targets, valid = prepare_metric_batch(
            logits, labels, valid_mask
        )
        return self.update(predictions, targets, valid, case_ids=case_ids)

    def compute(self) -> dict:
        if not self.records:
            raise RuntimeError("No KiPA22 metric records have been accumulated")
        case_class_scores = np.asarray(
            [
                [record["DSC_per_class"][KIPA22_LABELS[index]] for index in range(5)]
                for record in self.records
            ],
            dtype=np.float64,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            per_class = np.nanmean(case_class_scores, axis=0)
        result = {
            "DSC_per_class": {
                KIPA22_LABELS[class_id]: float(per_class[class_id])
                for class_id in range(self.num_classes)
            },
            "per_case": list(self.records),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for name, class_ids in KIPA22_GROUPS.items():
                result[name] = grouped_point_estimate(case_class_scores, class_ids)
        result["DSC"] = result["foreground_macro_DSC"]
        return result


