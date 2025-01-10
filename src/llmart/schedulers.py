#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import copy
import torch
from collections.abc import Callable
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import (
    ConstantLR,
    LinearLR,
    ExponentialLR,
    CosineAnnealingLR,
    MultiStepLR,
)
from torch.optim.lr_scheduler import _enable_get_lr_call, LRScheduler

from .optim import GreedyCoordinateGradient
from .config import SchedulerConf


class GCGScheduler(LRScheduler):
    """Base scheduler class for Greedy Coordinate Gradient optimizer.

    This scheduler will round to nearest integer with a minimum value of 1.

    Args:
        optimizer: A GreedyCoordinateGradient optimizer instance.
        var_name: Name of the optimizer parameter group variable to schedule.
    """

    def __init__(
        self,
        optimizer: GreedyCoordinateGradient,
        var_name: str,
    ):
        self.optimizer = optimizer
        self.var_name = var_name
        self.base_lrs = [
            copy.deepcopy(param_group[self.var_name])
            for param_group in optimizer.param_groups
        ]
        self.last_values = self.base_lrs
        self.last_epoch = 0

        # Get dummy optimizer and tensor
        self._dummy_param = torch.tensor([1.0], requires_grad=True)
        self._dummy_optimizer = SGD([self._dummy_param], lr=1.0)

    def _setup_reference_scheduler(
        self,
        scheduler_class: Callable[
            ..., ConstantLR | LinearLR | ExponentialLR | CosineAnnealingLR | MultiStepLR
        ],
        **kwargs,
    ):
        self._reference_scheduler = scheduler_class(self._dummy_optimizer, **kwargs)
        # Synchronize reference scheduler with GCG scheduler because of the step() called in init
        self._reference_scheduler.base_lrs = self.base_lrs
        self._reference_scheduler.last_epoch = self.last_epoch

    def get_new_values(self) -> list[int]:
        self._reference_scheduler.last_epoch += 1
        new_values = self._reference_scheduler._get_closed_form_lr()
        return [max(round(new_value), 1) for new_value in new_values]

    def get_last_lr(self) -> list[int] | list[float]:  # type: ignore[override]
        return self.last_values

    def step(self, loss: torch.Tensor):  # type: ignore[override]
        self.last_epoch += 1

        with _enable_get_lr_call(self):
            new_values = self.get_new_values()

        for idx, new_value in enumerate(new_values):
            self.optimizer.param_groups[idx][self.var_name] = new_value
        self.last_values = new_values


class LambdaInteger(GCGScheduler):
    """Scheduler that multiplies desired optimizer variable by a lambda.

    This scheduler will round to nearest integer with a minimum value of 1.

    Args:
        optimizer: A GreedyCoordinateGradient optimizer instance.
        var_name: Name of the optimizer parameter group variable to schedule.
        lr_lambda: Function that computes a multiplier given an epoch count.
    """

    def __init__(
        self,
        optimizer: GreedyCoordinateGradient,
        var_name: str,
        lr_lambda: Callable[[int], float],
    ):
        super().__init__(optimizer, var_name)
        self.lr_lambda = lr_lambda

    def get_new_values(self) -> list[int]:
        new_values = [
            max(round(base_lr * self.lr_lambda(self.last_epoch)), 1)
            for base_lr in self.base_lrs
        ]

        return new_values


class ConstantInteger(GCGScheduler):
    """Scheduler that scales desired optimizer variable for a specified number of steps.

    This scheduler will round to nearest integer with a minimum value of 1. See
    :class:`~torch.optim.lr_scheduler.ConstantLR` for more info.

    Args:
        optimizer: A GreedyCoordinateGradient optimizer instance.
        var_name: Name of the optimizer parameter group variable to schedule.
        factor: Number we multiply the parameter group variable until milestone.
        total_iters: Number of steps to scale parameter group variable.
    """

    def __init__(
        self,
        optimizer: GreedyCoordinateGradient,
        var_name: str,
        factor: float = 1.0,
        total_iters: int = 10,
    ):
        super().__init__(optimizer, var_name)
        self.factor = factor
        self.total_iters = total_iters
        self.last_epoch = -1  # Accounts for the upcoming step
        # PyTorch's ConstantLR will assert on exactly 1.0
        if self.factor > 0.0 and self.factor < 1.0:
            self._setup_reference_scheduler(
                ConstantLR, factor=factor, total_iters=total_iters
            )
            # Takes effect immediately
            self.step(torch.tensor(0.0))

    def get_new_values(self) -> list[int]:
        if self.factor == 1.0:
            new_values = self.last_values
        else:
            self._reference_scheduler.last_epoch += 1
            new_values = self._reference_scheduler._get_closed_form_lr()

        return [max(round(new_value), 1) for new_value in new_values]


class LinearInteger(GCGScheduler):
    """Scheduler that linearly scales the desired optimizer variable between two values.

    This scheduler will round to nearest integer with a minimum value of 1. See
    :class:`~torch.optim.lr_scheduler.ConstantLR` for more info.

    Args:
        optimizer: A GreedyCoordinateGradient optimizer instance.
        var_name: Name of the optimizer parameter group variable to schedule.
        start_factor: Number we multiply parameter group variable in first epoch.
        end_factor: Number we multiply parameter group variable at end of linear
            changing process.
        total_iters: Number of iterations for linear changing process.
    """

    def __init__(
        self,
        optimizer: GreedyCoordinateGradient,
        var_name: str,
        start_factor: float = 1.0,
        end_factor: float = 0.5,
        total_iters: int = 20,
    ):
        super().__init__(optimizer, var_name)
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        self.last_epoch = -1  # Accounts for the upcoming step
        self._setup_reference_scheduler(
            LinearLR,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=total_iters,
        )
        # Takes effect immediately
        self.step(torch.tensor(0.0))


class ExponentialInteger(GCGScheduler):
    """Scheduler that exponentially scales the desired optimizer variable.

    This scheduler will round to nearest integer with a minimum value of 1. See
    :class:`~torch.optim.lr_scheduler.ExponentialLR` for more info.

    Args:
        optimizer: A GreedyCoordinateGradient optimizer instance.
        var_name: Name of the optimizer parameter group variable to schedule.
        gamma: Multiplicative factor of parameter group variable.
    """

    def __init__(
        self,
        optimizer: GreedyCoordinateGradient,
        var_name: str,
        gamma: float = 1.0,
    ):
        super().__init__(optimizer, var_name)
        self.gamma = gamma
        self._setup_reference_scheduler(
            ExponentialLR,
            gamma=gamma,
        )


class CosineAnnealingInteger(GCGScheduler):
    """Scheduler that adjusts multiplicative factor following a cosine curve.

    This scheduler will round to nearest integer with a minimum value of 1. See
    :class:`~torch.optim.lr_scheduler.CosineAnnealingLR` for more info.

    Args:
        optimizer: A GreedyCoordinateGradient optimizer instance.
        var_name: Name of the optimizer parameter group variable to schedule.
        eta_min: Minimum parameter group variable value.
        T_max: Number of iterations for a full cycle.
    """

    def __init__(
        self,
        optimizer: GreedyCoordinateGradient,
        var_name: str,
        eta_min: float = 0.5,
        T_max: int = 20,
    ):
        super().__init__(optimizer, var_name)
        self.eta_min = eta_min
        self.T_max = T_max
        self._setup_reference_scheduler(
            CosineAnnealingLR,
            eta_min=eta_min,
            T_max=T_max,
        )


class MultiStepInteger(GCGScheduler):
    """Scheduler that changes optimizer variable with factors at specified milestones.

    This scheduler will round to nearest integer with a minimum value of 1. See
    :class:`~torch.optim.lr_scheduler.MultiStepLR` for more info.

    Args:
        optimizer: A GreedyCoordinateGradient optimizer instance.
        var_name: Name of the variable to schedule.
        milestones: List of epochs at which to decay values.
        gamma: Multiplicative factor at each milestone.
    """

    def __init__(
        self,
        optimizer: GreedyCoordinateGradient,
        var_name: str,
        milestones: list[int] = [20],
        gamma: float = 1.0,
    ):
        super().__init__(optimizer, var_name)
        self.gamma = gamma
        self.milestones = milestones
        self._setup_reference_scheduler(
            MultiStepLR,
            gamma=gamma,
            milestones=milestones,
        )


class ChangeOnPlateauInteger(GCGScheduler):
    """Scheduler that reduces desired optimizer variable when loss plateaus.

    Args:
        optimizer: A GreedyCoordinateGradient optimizer instance.
        var_name: Name of the variable to schedule.
        factor: Factor to multiply values by on plateau.
        patience: Number of epochs to wait for improvement.
        threshold: Minimum change to qualify as improvement.
        threshold_mode: How to measure improvement ('rel' or 'abs').
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.
    """

    def __init__(
        self,
        optimizer: GreedyCoordinateGradient,
        var_name: str,
        factor: float = 0.5,
        patience: int = 1000,
        threshold: float = 0.01,
        threshold_mode: str = "abs",
        min_value: int = 1,
        max_value: int = 8,
    ):
        super().__init__(optimizer, var_name)
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.min_value = min_value
        self.max_value = max_value

        self.num_bad_steps = 0
        # Best loss always assumes "min"
        self.best_loss = torch.inf

    def get_new_values(self) -> list[int]:
        new_values = [last_value * self.factor for last_value in self.last_values]

        return [
            min(max(round(new_value), self.min_value), self.max_value)
            for new_value in new_values
        ]

    def step(self, loss: torch.Tensor):  # type: ignore[override]
        self.last_epoch += 1

        # NOTE: (torch.inf / torch.inf) < 0.1 returns False as of PyTorch 2.5
        if self.threshold_mode == "rel" and (loss / self.best_loss) < self.threshold:
            self.num_bad_steps = 0
            self.best_loss = loss
            return

        if self.threshold_mode == "abs" and (loss + self.threshold) < self.best_loss:
            self.num_bad_steps = 0
            self.best_loss = loss
            return

        # Increment failed step counter
        self.num_bad_steps += 1

        if self.num_bad_steps == self.patience:
            with _enable_get_lr_call(self):
                new_values = self.get_new_values()

            for idx, new_value in enumerate(new_values):
                self.optimizer.param_groups[idx][self.var_name] = new_value
            self.last_values = new_values
            # Reset
            self.num_bad_steps = 0


def from_config(
    cfg: SchedulerConf,
    optimizer: Optimizer | GreedyCoordinateGradient,
) -> LRScheduler:
    """Creates a scheduler instance from configuration.

    Args:
        cfg: Scheduler configuration.
        optimizer: Optimizer to pass to scheduler.

    Returns:
        Configured scheduler instance.

    Raises:
        ValueError: If optimizer name is unknown or not supported.
    """

    if isinstance(optimizer, GreedyCoordinateGradient):
        if cfg.name == "constant":
            return ConstantInteger(optimizer, cfg.var_name, cfg.factor, cfg.total_iters)
        if cfg.name == "linear":
            return LinearInteger(
                optimizer,
                cfg.var_name,
                cfg.start_factor,
                cfg.end_factor,
                cfg.total_iters,
            )
        if cfg.name == "exponential":
            return ExponentialInteger(optimizer, cfg.var_name, cfg.gamma)
        if cfg.name == "cosine":
            return CosineAnnealingInteger(
                optimizer, cfg.var_name, cfg.eta_min, cfg.T_max
            )
        if cfg.name == "multistep":
            return MultiStepInteger(optimizer, cfg.var_name, cfg.milestones, cfg.gamma)
        if cfg.name == "plateau":
            return ChangeOnPlateauInteger(
                optimizer,
                cfg.var_name,
                cfg.factor,
                cfg.patience,
                cfg.threshold,
                cfg.threshold_mode,
                cfg.min_value,
                cfg.max_value,
            )

    if isinstance(optimizer, Optimizer):
        if cfg.name == "constant":
            return ConstantLR(optimizer, factor=1.0)

    raise ValueError(f"Invalid {cfg.name = } for soft prompt scheduler!")
