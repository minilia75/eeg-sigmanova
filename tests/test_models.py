import torch
import pytest

from eeg_sigmanova.models.cbramod import CBraMod
from eeg_sigmanova.models.eeg_simple_conv import EEGSimpleConv
from eeg_sigmanova.models.cbramod_classifier import CBraModBinaryClassifier


def test_cbramod_output_shape():
    model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=2, nhead=8)
    x = torch.randn(2, 16, 4, 200)
    y = model(x)
    assert y.shape == x.shape


def test_cbramod_with_mask():
    model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=2, nhead=8)
    x = torch.randn(2, 16, 4, 200)
    mask = torch.zeros(2, 16, 4, dtype=torch.long)
    mask[0, :3, :] = 1
    y = model(x, mask=mask)
    assert y.shape == x.shape


def test_shu_model_no_weights():
    model = CBraModBinaryClassifier(weights_path=None, n_channels=32, n_patches=4, d_model=200)
    x = torch.randn(2, 32, 4, 200)
    y = model(x)
    assert y.shape == (2,)


def test_eeg_simple_conv_output_shape():
    model = EEGSimpleConv(
        n_channels=32, n_classes=2, sfreq=800,
        fm=16, n_convs=2, resampling=128, kernel_size=8,
    )
    x = torch.randn(2, 32, 800)
    y = model(x)
    assert y.shape == (2, 2)


def test_eeg_simple_conv_with_subjects():
    model = EEGSimpleConv(
        n_channels=32, n_classes=2, sfreq=800,
        fm=16, n_convs=2, resampling=128, kernel_size=8,
        n_subjects=10,
    )
    x = torch.randn(2, 32, 800)
    task_logits, domain_logits = model(x)
    assert task_logits.shape == (2, 2)
    assert domain_logits.shape == (2, 10)
