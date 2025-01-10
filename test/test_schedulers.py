#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch
from .test_optim import make_onehot

from llmart import GreedyCoordinateGradient, LambdaInteger, ChangeOnPlateauInteger
from llmart.schedulers import (
    ConstantInteger,
    LinearInteger,
    ExponentialInteger,
    CosineAnnealingInteger,
    MultiStepInteger,
)

from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.optim.sgd import SGD


@pytest.fixture
def optimizer():
    return GreedyCoordinateGradient(
        [make_onehot(20, 512, dtype=torch.float, device="cpu", requires_grad=True)],
        n_tokens=15,
    )


def test_lambda(optimizer):
    scheduler = LambdaInteger(optimizer, "n_tokens", lambda step: 1 / step)
    scheduler.step(torch.tensor(0.0))  # 1
    assert optimizer.param_groups[0]["n_tokens"] == 15
    scheduler.step(torch.tensor(0.0))  # 2
    assert optimizer.param_groups[0]["n_tokens"] == 8
    scheduler.step(torch.tensor(0.0))  # 3
    assert optimizer.param_groups[0]["n_tokens"] == 5
    scheduler.step(torch.tensor(0.0))  # 4
    assert optimizer.param_groups[0]["n_tokens"] == 4
    scheduler.step(torch.tensor(0.0))  # 5
    assert optimizer.param_groups[0]["n_tokens"] == 3
    scheduler.step(torch.tensor(0.0))  # 6
    assert optimizer.param_groups[0]["n_tokens"] == 2
    scheduler.step(torch.tensor(0.0))  # 7
    assert optimizer.param_groups[0]["n_tokens"] == 2
    scheduler.step(torch.tensor(0.0))  # 8
    assert optimizer.param_groups[0]["n_tokens"] == 2
    scheduler.step(torch.tensor(0.0))  # 9
    assert optimizer.param_groups[0]["n_tokens"] == 2
    scheduler.step(torch.tensor(0.0))  # 10
    assert optimizer.param_groups[0]["n_tokens"] == 2
    scheduler.step(torch.tensor(0.0))  # 11
    assert optimizer.param_groups[0]["n_tokens"] == 1


def test_constant(optimizer):
    # https://github.com/pytorch/pytorch/blob/5f28c42746709eeea7e0632f676d28cdad408e6b/torch/optim/lr_scheduler.py#L637
    optimizer.param_groups[0]["n_swaps"] = 500
    scheduler = ConstantInteger(optimizer, "n_swaps", factor=0.5, total_iters=4)
    # Takes immediate effect
    assert optimizer.param_groups[0]["n_swaps"] == 250

    for _ in range(3):
        scheduler.step(torch.tensor(0.0))
        assert optimizer.param_groups[0]["n_swaps"] == 250

    scheduler.step(torch.tensor(0.0))
    assert optimizer.param_groups[0]["n_swaps"] == 500


def test_linear(optimizer):
    # https://github.com/pytorch/pytorch/blob/5f28c42746709eeea7e0632f676d28cdad408e6b/torch/optim/lr_scheduler.py#L713
    optimizer.param_groups[0]["n_swaps"] = 500
    scheduler = LinearInteger(
        optimizer, "n_swaps", start_factor=0.5, end_factor=1.0, total_iters=4
    )
    # Takes immediate effect
    assert optimizer.param_groups[0]["n_swaps"] == 250

    scheduler.step(torch.tensor(0.0))
    assert optimizer.param_groups[0]["n_swaps"] == 312

    scheduler.step(torch.tensor(0.0))
    assert optimizer.param_groups[0]["n_swaps"] == 375

    scheduler.step(torch.tensor(0.0))
    assert optimizer.param_groups[0]["n_swaps"] == 438

    for _ in range(3):
        scheduler.step(torch.tensor(0.0))
        assert optimizer.param_groups[0]["n_swaps"] == 500


def test_exponential(optimizer):
    optimizer.param_groups[0]["n_swaps"] = 500
    scheduler = ExponentialInteger(optimizer, "n_swaps", gamma=1.2)

    scheduler.step(torch.tensor(0.0))
    assert optimizer.param_groups[0]["n_swaps"] == 600

    scheduler.step(torch.tensor(0.0))
    assert optimizer.param_groups[0]["n_swaps"] == 720

    _dummy_var = torch.tensor(1.0, requires_grad=True)
    _dummy_optimizer = SGD([_dummy_var], lr=500.0)
    _reference_scheduler = ExponentialLR(_dummy_optimizer, gamma=1.2)
    _dummy_var.sum().backward()

    _dummy_optimizer.step()
    _reference_scheduler.step()
    assert _dummy_optimizer.param_groups[0]["lr"] == 600.0

    _dummy_optimizer.step()
    _reference_scheduler.step()
    assert _dummy_optimizer.param_groups[0]["lr"] == 720.0


def test_cosine(optimizer):
    optimizer.param_groups[0]["n_swaps"] = 500
    scheduler = CosineAnnealingInteger(optimizer, "n_swaps", eta_min=0.5, T_max=6)
    # Reference
    _dummy_var = torch.tensor(1.0, requires_grad=True)
    _dummy_optimizer = SGD([_dummy_var], lr=500.0)
    # Torch forgot to annotate "eta_min: float"
    _reference_scheduler = CosineAnnealingLR(_dummy_optimizer, eta_min=0.5, T_max=6)  # type: ignore
    _dummy_var.sum().backward()

    # Check a complete period
    for _ in range(2 * 6):
        scheduler.step(torch.tensor(0.0))
        _dummy_optimizer.step()
        _reference_scheduler.step()
        assert optimizer.param_groups[0]["n_swaps"] == max(
            round(_dummy_optimizer.param_groups[0]["lr"]), 1
        )


def test_multistep(optimizer):
    optimizer.param_groups[0]["n_swaps"] = 500
    scheduler = MultiStepInteger(optimizer, "n_swaps", milestones=[2, 3], gamma=0.7)

    scheduler.step(torch.tensor(0.0))  # After 1 step
    assert optimizer.param_groups[0]["n_swaps"] == 500

    scheduler.step(torch.tensor(0.0))  # After 2 steps, one reduction kicks in
    assert optimizer.param_groups[0]["n_swaps"] == 350

    scheduler.step(torch.tensor(0.0))  # After 3 steps, two reductions kick in
    assert optimizer.param_groups[0]["n_swaps"] == 245


def test_change_on_plateau(optimizer):
    patience = 10
    scheduler = ChangeOnPlateauInteger(
        optimizer,
        "n_tokens",
        patience=patience,
        max_value=15,
        threshold=0.1,
    )
    scheduler.step(torch.tensor(1.0))
    # Nothing good
    for _ in range(patience):
        scheduler.step(torch.tensor(1.0))
    assert optimizer.param_groups[0]["n_tokens"] == 8

    # Find better losses, but not sufficiently better
    for _ in range(patience):
        scheduler.step(torch.tensor(0.99))
    assert optimizer.param_groups[0]["n_tokens"] == 4

    # Find a better loss every 2 steps
    for idx in range(patience):
        scheduler.step(torch.tensor(1.0 - idx / (2 * patience)))
        scheduler.step(torch.tensor(1.0 - idx / (3 * patience)))
        assert optimizer.param_groups[0]["n_tokens"] == 4

    # Nothing good
    for _ in range(patience):
        scheduler.step(torch.tensor(1.0))
    assert optimizer.param_groups[0]["n_tokens"] == 2
    for _ in range(patience):
        scheduler.step(torch.tensor(1.0))
    assert optimizer.param_groups[0]["n_tokens"] == 1
