import logging
import pickle
import shutil
from pathlib import Path

import lmdb
import numpy as np
import scipy.io
from scipy import signal as sp_signal
from tqdm import tqdm

logger = logging.getLogger(__name__)

from eeg_sigmanova.constant import _LMDB_MAP_SIZE, _COMMIT_EVERY


def _lmdb_is_valid(lmdb_path: Path) -> bool:
    """Return True only if the directory exists AND the __keys__ sentinel is present."""
    if not lmdb_path.exists():
        return False
    try:
        env = lmdb.open(str(lmdb_path), readonly=True, lock=False, meminit=False)
        with env.begin() as txn:
            valid = txn.get(b"__keys__") is not None
        env.close()
        return valid
    except Exception:
        return False


def build_lmdb(
    mat_dir: Path,
    lmdb_path: Path,
    split: dict[str, list[int]],
    map_size: int = _LMDB_MAP_SIZE,
    commit_every: int = _COMMIT_EVERY,
) -> None:
    """Build an LMDB cache from raw .mat files.

    Each record: {'sample': np.ndarray(n_channels, n_patches, patch_size), 'label': int}.
    A '__keys__' sentinel maps split names to lists of record keys.
    """
    if _lmdb_is_valid(lmdb_path):
        logger.info(f"Valid LMDB found at {lmdb_path} — skipping build.")
        return

    if lmdb_path.exists():
        logger.warning(f"Incomplete LMDB found at {lmdb_path} — removing and rebuilding.")
        shutil.rmtree(lmdb_path)

    lmdb_path.mkdir(parents=True, exist_ok=True)
    db = lmdb.open(str(lmdb_path), map_size=map_size)
    txn = db.begin(write=True)

    subject_to_split = {s: name for name, subjects in split.items() for s in subjects}
    keys_index: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    pending = 0

    for mat_path in tqdm(sorted(mat_dir.glob("*.mat")), desc="Building LMDB"):
        parts = mat_path.stem.split("_")
        subj_id = int(parts[0].replace("sub-", ""))
        ses_id = parts[1]

        if subj_id not in subject_to_split:
            continue
        split_name = subject_to_split[subj_id]

        mat = scipy.io.loadmat(mat_path)
        eeg = mat["data"]        # (100, n_channels, n_points)
        labels = mat["labels"][0]  # (100,)

        eeg_rs = sp_signal.resample(eeg, 800, axis=2)          # → (100, n_channels, 800)
        eeg_rs = eeg_rs.reshape(eeg_rs.shape[0], 32, 4, 200)   # → (100, 32, 4, 200)

        for trial_idx, (sample, label) in enumerate(zip(eeg_rs, labels)):
            key = f"sub{subj_id:03d}-{ses_id}-trial{trial_idx:03d}"
            record = {"sample": sample.astype(np.float32), "label": int(label) - 1}
            txn.put(key.encode(), pickle.dumps(record))
            keys_index[split_name].append(key)
            pending += 1
            if pending % commit_every == 0:
                txn.commit()
                txn = db.begin(write=True)

    txn.put(b"__keys__", pickle.dumps(keys_index))
    txn.commit()
    db.close()

    total = sum(len(v) for v in keys_index.values())
    logger.info(f"LMDB written to {lmdb_path}")
    for name in ("train", "val", "test"):
        logger.info(f"  {name:5s}: {len(keys_index[name]):5d} trials")
    logger.info(f"  total: {total} trials")
