import copy
import logging
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eeg_sigmanova.training.evaluator import evaluate

__all__ = ["Trainer"]

logger = logging.getLogger(__name__)


class Trainer:
    """Generic trainer that tracks the best model checkpoint by val ROC-AUC.

    Args:
        scheduler_step: 'batch' for cosine-style schedulers stepped each batch
                        (CBraMod), 'epoch' for step-LR schedulers stepped each epoch
                        (EEGSimpleConv).
        use_sigmoid: Passed through to evaluate(). True for binary (CBraMod),
                     False for multi-class (EEGSimpleConv).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        clip_value: float = 1.0,
        scheduler_step: Literal["batch", "epoch"] = "batch",
        use_sigmoid: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.clip_value = clip_value
        self.scheduler_step = scheduler_step
        self.use_sigmoid = use_sigmoid

        self.history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_roc_auc": [],
            "val_pr_auc": [],
        }
        self.best_roc_auc: float = 0.0
        self.best_epoch: int = 0
        self.best_val_metrics: dict | None = None
        self._best_state: dict | None = None

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        epoch_losses = []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
            self.optimizer.step()
            if self.scheduler is not None and self.scheduler_step == "batch":
                self.scheduler.step()
            epoch_losses.append(loss.item())
        return float(np.mean(epoch_losses))

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> None:
        """Train for `epochs`, saving the best checkpoint by val ROC-AUC."""
        for epoch in range(1, epochs + 1):
            t0 = timer()
            train_loss = self._train_epoch(train_loader)

            if self.scheduler is not None and self.scheduler_step == "epoch":
                self.scheduler.step()

            val_m = evaluate(
                self.model, val_loader, self.device,
                criterion=self.criterion, use_sigmoid=self.use_sigmoid,
            )
            elapsed = (timer() - t0) / 60

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_m["loss"])
            self.history["val_acc"].append(val_m["acc"])
            self.history["val_roc_auc"].append(val_m["roc_auc"])
            self.history["val_pr_auc"].append(val_m["pr_auc"])

            is_best = val_m["roc_auc"] > self.best_roc_auc
            if is_best:
                self.best_roc_auc = val_m["roc_auc"]
                self.best_epoch = epoch
                self.best_val_metrics = val_m
                self._best_state = copy.deepcopy(self.model.state_dict())

            lr_now = self.optimizer.param_groups[0]["lr"]
            flag = "  <- best" if is_best else ""
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train {train_loss:.4f} | "
                f"val {val_m['loss']:.4f} | "
                f"val acc {val_m['acc']:.4f} | "
                f"val ROC-AUC {val_m['roc_auc']:.4f} | "
                f"val PR-AUC {val_m['pr_auc']:.4f} | "
                f"lr {lr_now:.2e} | "
                f"{elapsed:.1f} min{flag}"
            )

        logger.info(f"\nBest validation ROC-AUC {self.best_roc_auc:.4f} at epoch {self.best_epoch}")

    def load_best(self) -> None:
        """Restore the best checkpoint weights into the model in-place."""
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
