#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn.functional as F

from llmart import CausalLMLoss


def test_causalloss():
    logits = torch.tensor(
        [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0], [0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([-1, 2, 2, 2])  # correct, wrong, wrong

    loss_fn = CausalLMLoss(F.nll_loss, reduction="mean")

    loss = loss_fn(logits, labels)
    loss.backward()

    # gradient only on target logits
    grad = torch.zeros_like(logits.grad)  # type: ignore
    grad[0, 2] = -1 / (len(labels) - 1)
    grad[1, 2] = -1 / (len(labels) - 1)
    grad[2, 2] = -1 / (len(labels) - 1)

    torch.testing.assert_close(logits.grad, grad)


def test_hardmax():
    logits = torch.tensor(
        [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0], [0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([-1, 2, 2, 2])  # ignored, correct, wrong, wrong

    loss_fn = CausalLMLoss(F.nll_loss, reduction="hardmax")

    loss = loss_fn(logits, labels)
    loss.backward()

    # first two logits should only have gradient
    grad = torch.zeros_like(logits.grad)  # type: ignore
    grad[0, 2] = -1 / (len(labels) - 1)
    grad[1, 2] = -1 / (len(labels) - 1)

    torch.testing.assert_close(logits.grad, grad)


def test_mellowmax_large_alpha():
    logits = torch.tensor(
        [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0], [0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([-1, 2, 2, 2])  # ignored, correct, wrong, wrong

    loss_fn = CausalLMLoss(F.nll_loss, reduction="mellowmax", mm_alpha=1000)

    loss = loss_fn(logits, labels)
    loss.backward()

    # gradient only on logit that is maximally incorrect
    grad = torch.zeros_like(logits.grad)  # type: ignore
    grad[1, 2] = -1

    torch.testing.assert_close(logits.grad, grad)


def test_mellowmax_small_alpha():
    logits = torch.tensor(
        [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0], [0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([-1, 2, 2, 2])  # ignored, correct, wrong, wrong

    loss_fn = CausalLMLoss(F.nll_loss, reduction="mellowmax", mm_alpha=1e-8)

    loss = loss_fn(logits, labels)
    loss.backward()

    # gradient is average of target logits
    grad = torch.zeros_like(logits.grad)  # type: ignore
    grad[0, 2] = -1 / (len(labels) - 1)
    grad[1, 2] = -1 / (len(labels) - 1)
    grad[2, 2] = -1 / (len(labels) - 1)

    torch.testing.assert_close(logits.grad, grad)
