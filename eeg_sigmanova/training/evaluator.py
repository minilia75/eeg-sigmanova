import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader

__all__ = ["evaluate"]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module | None = None,
    use_sigmoid: bool = True,
) -> dict:
    """Compute classification metrics on a data loader.

    Args:
        use_sigmoid: True for binary classification (BCEWithLogitsLoss / CBraMod).
                     False for multi-class (CrossEntropyLoss / EEGSimpleConv).

    Returns:
        Dict with keys: loss, acc (balanced), roc_auc, pr_auc, cm (confusion matrix).
    """
    model.eval()
    truths, preds, scores, losses = [], [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        if criterion is not None:
            losses.append(criterion(logits, y).item())

        if use_sigmoid:
            prob = torch.sigmoid(logits)
            pred = (prob > 0.5).long()
        else:
            prob = F.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(dim=1)

        truths.extend(y.long().cpu().tolist())
        preds.extend(pred.cpu().tolist())
        scores.extend(prob.cpu().tolist())

    truths = np.array(truths)
    preds = np.array(preds)
    scores = np.array(scores)

    precision, recall, _ = precision_recall_curve(truths, scores, pos_label=1)
    return {
        "loss": float(np.mean(losses)) if losses else None,
        "acc": balanced_accuracy_score(truths, preds),
        "roc_auc": roc_auc_score(truths, scores),
        "pr_auc": auc(recall, precision),
        "cm": confusion_matrix(truths, preds),
    }
