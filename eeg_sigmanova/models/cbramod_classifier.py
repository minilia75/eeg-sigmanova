from pathlib import Path

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from eeg_sigmanova.models.cbramod import CBraMod

__all__ = ["CBraModBinaryClassifier"]


class CBraModBinaryClassifier(nn.Module):
    """CBraMod backbone with a binary classification head.

    The backbone's proj_out is replaced with nn.Identity() after weight loading
    so the encoder output (B, C, S, D) is passed directly to the head.

    Args:
        weights_path: Path to pretrained CBraMod weights. Pass None to skip
                      weight loading (useful for testing with random weights).
        classifier: Head architecture. One of 'all_patch_reps' (default, 3-layer MLP),
                    'all_patch_reps_twolayer', 'all_patch_reps_onelayer', 'avgpooling_patch_reps'.
        n_channels: Number of EEG channels (default 32 for SHU-MI).
        n_patches:  Number of temporal patches per channel (default 4).
        d_model:    Transformer hidden dimension (default 200).
    """

    def __init__(
        self,
        weights_path: Path | None,
        classifier: str = "all_patch_reps",
        dropout: float = 0.1,
        device: str = "cpu",
        n_channels: int = 32,
        n_patches: int = 4,
        d_model: int = 200,
    ):
        super().__init__()
        self.backbone = CBraMod(
            in_dim=d_model,
            out_dim=d_model,
            d_model=d_model,
            dim_feedforward=4 * d_model,
            seq_len=30,
            n_layer=12,
            nhead=8,
        )
        if weights_path is not None:
            state = torch.load(weights_path, map_location=device, weights_only=True)
            self.backbone.load_state_dict(state)
        self.backbone.proj_out = nn.Identity()

        flat = n_channels * n_patches * d_model

        if classifier == "avgpooling_patch_reps":
            self.head = nn.Sequential(
                Rearrange("b c s d -> b d c s"),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(d_model, 1),
                Rearrange("b 1 -> (b 1)"),
            )
        elif classifier == "all_patch_reps_onelayer":
            self.head = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(flat, 1),
                Rearrange("b 1 -> (b 1)"),
            )
        elif classifier == "all_patch_reps_twolayer":
            self.head = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(flat, d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
                Rearrange("b 1 -> (b 1)"),
            )
        elif (
            classifier == "all_patch_reps"
        ):  # all_patch_reps — 3-layer MLP (paper default)
            self.head = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(flat, 4 * d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
                Rearrange("b 1 -> (b 1)"),
            )
        else:
            raise ValueError(
                f"Unknown classifier {classifier!r}. "
                "Expected one of: 'all_patch_reps', 'all_patch_reps_twolayer', "
                "'all_patch_reps_onelayer', 'avgpooling_patch_reps'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
