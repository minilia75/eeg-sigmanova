import torch
import torch.nn as nn
from torch import Tensor
from torchaudio.transforms import Resample

__all__ = ["EEGSimpleConv"]


class EEGSimpleConv(nn.Module):
    """EEGSimpleConv (Kostas et al., 2022).

    Input:  (batch, n_channels, n_samples) — raw signal at sfreq Hz
    Output: (batch, n_classes) logits, or ((batch, n_classes), (batch, n_subjects))
            when n_subjects is set (domain adaptation head).
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        sfreq: int,
        fm: int = 64,
        n_convs: int = 4,
        resampling: int = 128,
        kernel_size: int = 8,
        n_subjects: int | None = None,
    ):
        super().__init__()
        self.rs = Resample(orig_freq=sfreq, new_freq=resampling)
        self.conv = nn.Conv1d(n_channels, fm, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm1d(fm)

        blocks = []
        in_fm = fm
        for i in range(n_convs):
            out_fm = int(1.414 * in_fm) if i > 0 else in_fm
            blocks.append(nn.Sequential(
                nn.Conv1d(in_fm, out_fm, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(out_fm),
                nn.MaxPool1d(2),
                nn.ReLU(),
                nn.Conv1d(out_fm, out_fm, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(out_fm),
                nn.ReLU(),
            ))
            in_fm = out_fm

        self.blocks = nn.ModuleList(blocks)
        self.fc = nn.Linear(in_fm, n_classes)
        self.fc2 = nn.Linear(in_fm, n_subjects) if n_subjects else None

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        y = torch.relu(self.bn(self.conv(self.rs(x.contiguous()))))
        for block in self.blocks:
            y = block(y)
        y = y.mean(dim=2)
        return (self.fc(y), self.fc2(y)) if self.fc2 else self.fc(y)
