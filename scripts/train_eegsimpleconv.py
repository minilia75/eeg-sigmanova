"""Train EEGSimpleConv on the SHU motor-imagery dataset.

Usage:
    uv run python scripts/train_eegsimpleconv.py
    uv run python scripts/train_eegsimpleconv.py training.epochs=100 model.fm=128
    uv run python scripts/train_eegsimpleconv.py training.lr=5e-4
"""

import logging
import sys
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

from eeg_sigmanova.data.dataset import make_raw_loaders
from eeg_sigmanova.data.preprocessing import build_lmdb
from eeg_sigmanova.models.eeg_simple_conv import EEGSimpleConv
from eeg_sigmanova.training.evaluator import evaluate
from eeg_sigmanova.training.trainer import Trainer
from eeg_sigmanova.utils import get_device, set_seed

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../configs", config_name="train_eegsimpleconv"
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = get_device(cfg.training.device)
    set_seed(cfg.training.seed)
    log.info(f"Device: {device}")
    log.info(
        f"Model: EEGSimpleConv  fm={cfg.model.fm}  n_convs={cfg.model.n_convs}  resampling={cfg.model.resampling}  kernel={cfg.model.kernel_size}"
    )

    build_lmdb(
        mat_dir=Path(cfg.data.mat_dir),
        lmdb_path=Path(cfg.data.lmdb_path),
        split={
            "train": list(cfg.data.train_subjects),
            "val": list(cfg.data.val_subjects),
            "test": list(cfg.data.test_subjects),
        },
    )

    loaders, mean, std = make_raw_loaders(
        lmdb_path=Path(cfg.data.lmdb_path),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        n_channels=cfg.data.n_channels,
        device=str(device),
    )
    log.info(f"Channel mean range: [{mean.min():.3f}, {mean.max():.3f}]")
    log.info(f"Channel std  range: [{std.min():.3f}, {std.max():.3f}]")
    for split, dl in loaders.items():
        log.info(f"{split:5s}: {len(dl.dataset):5d} trials — {len(dl)} batches")

    model = EEGSimpleConv(
        n_channels=cfg.model.n_channels,
        n_classes=cfg.model.n_classes,
        sfreq=cfg.model.sfreq,
        fm=cfg.model.fm,
        n_convs=cfg.model.n_convs,
        resampling=cfg.model.resampling,
        kernel_size=cfg.model.kernel_size,
        n_subjects=cfg.model.n_subjects,
    ).to(device)
    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=(4 * cfg.training.epochs) // 5,
        gamma=0.1,
    )
    criterion = nn.CrossEntropyLoss().to(device)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=str(device),
        scheduler=scheduler,
        clip_value=cfg.training.clip_value,
        scheduler_step="epoch",
        use_sigmoid=False,
    )
    trainer.fit(loaders["train"], loaders["val"], epochs=cfg.training.epochs)
    trainer.load_best()

    test_m = evaluate(model, loaders["test"], str(device), use_sigmoid=False)
    log.info("=" * 50)
    log.info("TEST RESULTS  (subjects 21–25)")
    log.info("=" * 50)
    log.info(f"  Balanced Accuracy : {test_m['acc']:.4f}")
    log.info(f"  ROC-AUC           : {test_m['roc_auc']:.4f}")
    log.info(f"  PR-AUC            : {test_m['pr_auc']:.4f}")

    ckpt = (
        Path(cfg.training.checkpoint_dir)
        / f"eegsimpleconv_epoch{trainer.best_epoch}_roc{trainer.best_roc_auc:.4f}.pth"
    )
    trainer.save(ckpt)


if __name__ == "__main__":
    main()
