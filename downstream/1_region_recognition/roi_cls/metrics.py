"""Metrics for frozen CARE ROI classification."""

from torchmetrics import AUROC, MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification.f_beta import F1Score


def build_linear_metric(num_classes):
    """Return the original macro multiclass metrics without bootstrapping."""
    num_classes = int(num_classes)
    return MetricCollection(
        {
            "top1 acc": MulticlassAccuracy(
                top_k=1, num_classes=num_classes, average="macro"
            ),
            "f1_score": F1Score(
                num_classes=num_classes, average="macro", task="multiclass"
            ),
            "AUC": AUROC(
                num_classes=num_classes, average="macro", task="multiclass"
            ),
        }
    )
