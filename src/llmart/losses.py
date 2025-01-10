#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import math
import torch
from typing import Any
from collections.abc import Callable, Mapping
from functools import partial
from torch.nn import functional as F

from .config import CausalLMLossConf


def ranking_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.0,
    rank: int = 0,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Computes ranking loss between target and rank-based predictions.

    Uses :class:`~torch.nn.MarginRankingLoss` to measure whether input targets have larger value
    than inputs of specified rank taking care to ignore target rank.

    Args:
        input: Log probabilities of shape (batch_size, num_classes).
        target: Target indices of shape (batch_size,).
        margin: Minimum difference between target and rank probabilities.
        rank: Which rank to compare target against (0 = most likely).
        ignore_index: Target indices to ignore in loss computation.
        reduction: Reduction method ('none', 'mean', 'sum').

    Returns:
        torch.Tensor: Computed ranking loss based on specified reduction.

    Raises:
        ValueError: If reduction method is invalid.
    """

    orig_target = target
    mask = target != ignore_index
    input = input[mask].contiguous()
    target = target[mask].contiguous()

    target_lls = -F.nll_loss(input, target, reduction="none")

    # make targets less likely then sort to get ranks
    ranks = input.scatter(
        -1, target[..., None], torch.full_like(target[..., None].float(), -torch.inf)
    ).argsort(descending=True)
    rank_target = ranks[:, rank]
    rank_lls = -F.nll_loss(input, rank_target, reduction="none")

    # make sure lls are same size as original targets
    if reduction == "none":
        index = torch.where(mask)[0]
        target_lls = torch.zeros_like(orig_target, dtype=target_lls.dtype).scatter(
            -1, index, target_lls
        )
        rank_lls = torch.zeros_like(orig_target, dtype=rank_lls.dtype).scatter(
            -1, index, rank_lls
        )

    # make targets more likely than ranks
    return F.margin_ranking_loss(
        target_lls,
        rank_lls,
        target=torch.ones_like(target_lls),
        margin=margin,
        reduction=reduction,
    )


class CausalLMLoss(torch.nn.Module):
    """Loss module for causal language modeling tasks.

    This module shifts the labels and truncates the logits before passing them to
    the specified loss function. It adds support for hardmax and mellowmax reductions.
    A hardmax reduction has loss on up to (left-to-right) the specified number of
    incorrect predictions. A mellowmax reduction reduces to the max input as alpha
    increases and to the average as alpha goes to 0.

    Args:
        loss_fn: Optional custom loss function to use instead of cross entropy.
        ignore_index: Target indices to ignore in loss computation.
        reduction: Reduction method ('none', 'mean', 'sum', 'hardmax', 'mellowmax').
        hm_wrong: Maximum number of wrong predictions allowed for hardmax reduction.
        mm_alpha: Temperature parameter for mellowmax reduction.
        **kwargs: Additional arguments passed to loss_fn.
    """

    def __init__(
        self,
        loss_fn: Callable | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        hm_wrong: int = 1,
        mm_alpha: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        self.loss_fn = loss_fn
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.mm_alpha = mm_alpha
        self.hm_wrong = hm_wrong
        self.kwargs = kwargs

    def forward(
        self,
        inputs_or_logits: Mapping[str, Any] | torch.Tensor,
        labels: torch.LongTensor,
    ) -> torch.Tensor:
        # Return model loss if wanted
        if (
            self.loss_fn is None
            and isinstance(inputs_or_logits, Mapping)
            and "loss" in inputs_or_logits
        ):
            return inputs_or_logits["loss"]

        # Retrieve logits
        if isinstance(inputs_or_logits, Mapping):
            logits = inputs_or_logits["logits"]
        elif isinstance(inputs_or_logits, torch.Tensor):
            logits = inputs_or_logits
        else:
            raise ValueError(
                "inputs_or_logits must be Mapping with logits or torch.Tensor"
            )

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        # Mask out ignore index and flatten
        valid_mask = shift_labels != self.ignore_index
        shift_logits = shift_logits[valid_mask, :].contiguous().float()
        shift_labels = shift_labels[valid_mask].contiguous()

        # Default to xent in cases where loss_fn isn't specified
        loss_fn = partial(
            self.loss_fn or F.cross_entropy,
            shift_logits,
            shift_labels,
            ignore_index=self.ignore_index,
            **self.kwargs,
        )
        if self.reduction.startswith("hardmax"):
            outputs = loss_fn(reduction="none")
            is_wrong = shift_labels != shift_logits.argmax(-1)
            loss = _hardmax_reduction(
                outputs, is_wrong, max_wrong=self.hm_wrong, reduction="mean"
            )
        elif self.reduction.startswith("mellowmax"):
            outputs = loss_fn(reduction="none")
            loss = _mellowmax_reduction(outputs, alpha=self.mm_alpha, reduction="mean")
        else:
            loss = loss_fn(reduction=self.reduction)
        return loss


def _mellowmax_reduction(
    input: torch.Tensor, alpha: float = 1.0, reduction: str = "mean"
) -> torch.Tensor:
    if reduction == "sum":
        loss = torch.logsumexp(alpha * input, -1)
    elif reduction == "mean":
        loss = torch.logsumexp(alpha * input, -1) - math.log(len(input))
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    return 1 / alpha * loss


def _hardmax_reduction(
    input, is_wrong, *, reduction: str = "mean", max_wrong: int = 1
) -> torch.Tensor:
    # Up to max-wrong logits to have loss, otherwise ignore that loss by detaching
    cumulative_wrongs = torch.cumsum(is_wrong, dim=0)
    ignore_mask = cumulative_wrongs > max_wrong

    loss = torch.where(ignore_mask, input.detach(), input)
    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    return loss


def from_config(cfg: CausalLMLossConf) -> CausalLMLoss:
    """Creates CausalLMLoss instance from configuration.

    Args:
        cfg: Configuration object containing loss parameters.

    Returns:
        CausalLMLoss: Configured loss module.

    Raises:
        ValueError: If loss configuration is invalid.
    """

    loss_fn = partial(
        CausalLMLoss,
        reduction=cfg.reduction,
        hm_wrong=cfg.hm_wrong,
        mm_alpha=cfg.mm_alpha,
    )
    if cfg.name == "model":
        return loss_fn()

    if cfg.name == "xent":
        return loss_fn(F.cross_entropy)

    if cfg.name == "logit":
        return loss_fn(F.nll_loss)

    if cfg.name == "ranking":
        assert cfg.margin is not None
        assert cfg.rank is not None
        return loss_fn(
            partial(ranking_loss, margin=cfg.margin, rank=cfg.rank),
        )

    raise ValueError(f"Unknown loss cfg: {cfg}")
