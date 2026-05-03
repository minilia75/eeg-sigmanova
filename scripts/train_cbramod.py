"""Fine-tune CBraMod on the SHU motor-imagery dataset.

Usage:
    uv run python scripts/train_cbramod.py
    uv run python scripts/train_cbramod.py training.epochs=100
    uv run python scripts/train_cbramod.py model.finetune_mode=full training.lr_backbone=1e-5
"""
import logging
import sys
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

from eeg_sigmanova.data.dataset import make_patch_loaders
from eeg_sigmanova.data.preprocessing import build_lmdb
from eeg_sigmanova.models.cbramod_classifier import CBraModBinaryClassifier
from eeg_sigmanova.training.evaluator import evaluate
from eeg_sigmanova.training.trainer import Trainer
from eeg_sigmanova.utils import get_device, set_seed

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train_cbramod")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    device = get_device(cfg.training.device)
    set_seed(cfg.training.seed)
    log.info(f"Device: {device}  |  fine-tune mode: {cfg.model.finetune_mode}")

    weights_path = Path(cfg.model.weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if not weights_path.exists():
        log.info("Downloading pretrained weights from HuggingFace...")
        hf_hub_download(
            repo_id="weighting666/CBraMod",
            filename="pretrained_weights.pth",
            local_dir=str(weights_path.parent),
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

    loaders = make_patch_loaders(
        Path(cfg.data.lmdb_path),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        device=str(device),
        scale=cfg.data.scale,
    )
    for split, dl in loaders.items():
        log.info(f"{split:5s}: {len(dl.dataset):5d} trials — {len(dl)} batches")

    model = CBraModBinaryClassifier(
        weights_path=weights_path,
        classifier=cfg.model.classifier,
        dropout=cfg.model.dropout,
        device=str(device),
        n_channels=cfg.data.n_channels,
        n_patches=cfg.data.n_patches,
        d_model=cfg.model.d_model,
    ).to(device)

    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = (cfg.model.finetune_mode == "full")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable params: {trainable:,} / {total:,}  ({100 * trainable / total:.1f}%)")

    if cfg.model.finetune_mode == "head_only":
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": cfg.training.lr_head}]
    else:
        param_groups = [
            {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": cfg.training.lr_backbone},
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": cfg.training.lr_head},
        ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs * len(loaders["train"]),
        eta_min=1e-6,
    )
    criterion = nn.BCEWithLogitsLoss().to(device)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=str(device),
        scheduler=scheduler,
        clip_value=cfg.training.clip_value,
        scheduler_step="batch",
        use_sigmoid=True,
    )
    trainer.fit(loaders["train"], loaders["val"], epochs=cfg.training.epochs)
    trainer.load_best()

    test_m = evaluate(model, loaders["test"], str(device))
    log.info("=" * 50)
    log.info("TEST RESULTS  (subjects 21–25)")
    log.info("=" * 50)
    log.info(f"  Balanced Accuracy : {test_m['acc']:.4f}")
    log.info(f"  ROC-AUC           : {test_m['roc_auc']:.4f}")
    log.info(f"  PR-AUC            : {test_m['pr_auc']:.4f}")

    ckpt = Path(cfg.training.checkpoint_dir) / f"shu_mi_epoch{trainer.best_epoch}_roc{trainer.best_roc_auc:.4f}.pth"
    trainer.save(ckpt)


if __name__ == "__main__":
    main()
