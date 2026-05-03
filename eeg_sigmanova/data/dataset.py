import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SHUPatchDataset(Dataset):
    """LMDB-backed dataset returning (C, S, P) patched EEG and float labels.

    Used with CBraMod / BCEWithLogitsLoss. Labels are float.
    The LMDB environment is opened lazily inside each DataLoader worker
    to avoid sharing a forked file descriptor across processes.
    """

    def __init__(self, lmdb_path: Path, mode: str, scale: float = 100.0):
        self._path = str(lmdb_path)
        self._lmdb_env: lmdb.Environment | None = None
        self._scale = scale

        env = lmdb.open(self._path, readonly=True, lock=False, meminit=False)
        with env.begin() as txn:
            raw = txn.get(b"__keys__")
            if raw is None:
                raise RuntimeError(
                    f"LMDB at '{self._path}' is missing the '__keys__' index. "
                    "Re-run build_lmdb() to rebuild."
                )
            self.keys: list[str] = pickle.loads(raw)[mode]
        env.close()

    def __len__(self) -> int:
        return len(self.keys)

    def _get_env(self) -> lmdb.Environment:
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                self._path, readonly=True, lock=False, readahead=True, meminit=False
            )
        return self._lmdb_env

    def __getitem__(self, idx: int) -> tuple[np.ndarray, float]:
        with self._get_env().begin(write=False) as txn:
            record = pickle.loads(txn.get(self.keys[idx].encode()))
        return record["sample"] / self._scale, float(record["label"])

    @staticmethod
    def collate(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.stack([b[0] for b in batch])).float()
        y = torch.tensor([b[1] for b in batch]).float()
        return x, y


class SHURawDataset(Dataset):
    """LMDB-backed dataset returning (C, T) raw EEG and long labels.

    Used with EEGSimpleConv / CrossEntropyLoss. Labels are long.
    Patches stored as (C, S, P) are reshaped to (C, S*P) at read time.
    Optional per-channel z-score normalisation applied when mean/std provided.
    """

    def __init__(
        self,
        lmdb_path: Path,
        mode: str,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        self._path = str(lmdb_path)
        self._lmdb_env: lmdb.Environment | None = None
        self.mean = mean
        self.std = std

        env = lmdb.open(self._path, readonly=True, lock=False, meminit=False)
        with env.begin() as txn:
            raw = txn.get(b"__keys__")
            if raw is None:
                raise RuntimeError(
                    f"LMDB at '{self._path}' is missing '__keys__'. "
                    "Re-run build_lmdb() to rebuild."
                )
            self.keys: list[str] = pickle.loads(raw)[mode]
        env.close()

    def __len__(self) -> int:
        return len(self.keys)

    def _get_env(self) -> lmdb.Environment:
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                self._path, readonly=True, lock=False, readahead=True, meminit=False
            )
        return self._lmdb_env

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        with self._get_env().begin(write=False) as txn:
            record = pickle.loads(txn.get(self.keys[idx].encode()))
        x = record["sample"].reshape(record["sample"].shape[0], -1).astype(np.float32)
        if self.mean is not None:
            x = (x - self.mean[:, None]) / (self.std[:, None] + 1e-8)
        return x, int(record["label"])

    @staticmethod
    def collate(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.stack([b[0] for b in batch])).float()
        y = torch.tensor([b[1] for b in batch]).long()
        return x, y


def compute_channel_stats(
    lmdb_path: Path,
    n_channels: int,
    batch_size: int = 256,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std from the training split (online, O(1) memory)."""
    ds = SHURawDataset(lmdb_path, "train")
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=SHURawDataset.collate,
        num_workers=num_workers,
    )
    sum_x = torch.zeros(n_channels)
    sum_x2 = torch.zeros(n_channels)
    n_pts = 0
    for x, _ in dl:
        sum_x += x.sum(dim=(0, 2))
        sum_x2 += (x**2).sum(dim=(0, 2))
        n_pts += x.shape[0] * x.shape[2]
    mean = (sum_x / n_pts).numpy()
    std = ((sum_x2 / n_pts - torch.from_numpy(mean) ** 2).clamp(min=0).sqrt()).numpy()
    return mean, std


def make_patch_loaders(
    lmdb_path: Path,
    batch_size: int,
    num_workers: int,
    device: str = "cpu",
    scale: float = 100.0,
) -> dict[str, DataLoader]:
    """DataLoaders for CBraMod (patched input, float labels)."""
    return {
        mode: DataLoader(
            SHUPatchDataset(lmdb_path, mode, scale=scale),
            batch_size=batch_size,
            shuffle=(mode == "train"),
            collate_fn=SHUPatchDataset.collate,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
            persistent_workers=(num_workers > 0),
        )
        for mode in ("train", "val", "test")
    }


def make_raw_loaders(
    lmdb_path: Path,
    batch_size: int,
    num_workers: int,
    n_channels: int = 32,
    device: str = "cpu",
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[dict[str, DataLoader], np.ndarray, np.ndarray]:
    """DataLoaders for EEGSimpleConv (raw 1-D input, long labels) with z-score normalisation.

    Returns (loaders, mean, std) so callers can inspect or persist the normalisation stats.
    Pass pre-computed mean/std to skip the training-set scan.
    """
    if mean is None or std is None:
        mean, std = compute_channel_stats(lmdb_path, n_channels, num_workers=num_workers)

    loaders = {
        mode: DataLoader(
            SHURawDataset(lmdb_path, mode, mean=mean, std=std),
            batch_size=batch_size,
            shuffle=(mode == "train"),
            collate_fn=SHURawDataset.collate,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
            persistent_workers=(num_workers > 0),
        )
        for mode in ("train", "val", "test")
    }
    return loaders, mean, std
