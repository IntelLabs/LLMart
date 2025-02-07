#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import torch
import bisect
import inspect
import warnings
from operator import itemgetter
from collections import UserDict
from collections.abc import Callable
from accelerate.state import PartialState
from accelerate.utils import gather, reduce, pad_across_processes
from accelerate.utils import broadcast_object_list, tqdm
from torch.optim.optimizer import Optimizer, ParamsT  # type: ignore[reportPrivateImportUsage]
from torch.optim.sgd import SGD  # type: ignore[reportPrivateImportUsage]
from torch.optim.adam import Adam  # type: ignore[reportPrivateImportUsage]

from llmart.config import OptimConf
from llmart.pickers import pick_coord_topk, pick_global_topk, pick_unique_swaps
from llmart.samplers import sample_coord_randk, sample_swaps


class Coordinate(UserDict[int, int]):
    """A dictionary-like class for storing coordinate mappings.

    Stores index-to-value mappings for coordinates and provides conversion
    utilities to/from tensors.
    """

    @property
    def index(self) -> list[int]:
        return list(self.keys())

    @property
    def value(self) -> list[int]:
        return list(self.values())

    def to_tensor(self, device: str) -> torch.Tensor:
        return torch.tensor(
            [self.index, self.value], device=device
        ).T  # Index coordinates on row axis

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "Coordinate":
        if len(tensor.shape) == 1:
            return cls(enumerate(tensor.tolist()))
        else:
            return cls(zip(tensor[..., 0].tolist(), tensor[..., 1].tolist()))


class GreedyCoordinateGradient(Optimizer):
    """An optimizer implementing greedy coordinate gradient descent.

    Args:
        params: Parameters to optimize.
        negative_only: If True, only consider negative gradients.
        coord_randk: Number of random dictionary entries to sample per token.
        coord_topk: Number of top dictionary entries to select per token.
        global_topk: Number of top dictionary entries to select globally.
        n_tokens: Number of tokens to swap at once.
        n_swaps: Maximum number of swaps to attempt.
        n_buffers: Size of the buffer for storing best coordinates.
        ignore_curr_marginals: If True, ignore current token positions.
        ignored_values: Tensor of dictionary entries to ignore.
        embedding: Token embedding matrix.
        world_size: Number of distributed processes.

    Raises:
        ValueError: If params or embedding configuration is invalid.
    """

    def __init__(
        self,
        params: ParamsT,
        negative_only: bool = False,
        coord_randk: int = 0,
        coord_topk: int = 256,
        global_topk: int = 0,
        n_tokens: int = 20,
        n_swaps: int = 1024,
        n_buffers: int = 1,
        ignore_curr_marginals: bool = False,
        ignored_values: torch.Tensor | None = None,
        embedding: torch.nn.Module | None = None,
        world_size: int = 1,
    ):
        if ignored_values is None:
            ignored_values = torch.LongTensor()

        defaults = dict(
            negative_only=negative_only,
            coord_randk=coord_randk,
            coord_topk=coord_topk,
            global_topk=global_topk,
            n_tokens=n_tokens,
            n_swaps=n_swaps,
            n_buffers=max(n_buffers, 1),
            ignore_curr_marginals=ignore_curr_marginals,
            world_size=world_size,
            local_swap_count=torch.tensor(0),
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                f"{self.__class__.__name__} doesn't support per-parameter options (parameter groups)"
            )
        if len(self.param_groups[0]["params"]) != 1:
            raise ValueError(
                f"{self.__class__.__name__} doesn't support more than 1 parameter"
            )
        if len(self.param_groups[0]["params"][0].shape) != 2:
            raise ValueError(
                f"{self.__class__.__name__} requires 2-dimensional parameters"
            )
        if embedding is not None and not isinstance(embedding, torch.nn.Embedding):
            raise ValueError("embedding must be an nn.Embedding")

        if (
            embedding is not None
            and self.param_groups[0]["params"][0].shape[-1] != embedding.embedding_dim
        ):
            raise ValueError(
                f"Parameter must have same dimension as embedding ({embedding.embedding_dim})!"
            )

        if (
            embedding is None
            and (self.param_groups[0]["params"][0].sum(dim=-1) != 1.0).any()
        ):
            raise ValueError(
                f"{self.__class__.__name__} only works with 1-hot encoded parameters when embedding is None"
            )

        self._update_hyperparams()

        self._ignored_values = ignored_values
        self._embedding = embedding
        self._replacements: list[Coordinate] | None = None

    def _update_hyperparams(self):
        self._coord_topk = self.param_groups[0]["coord_topk"]
        self._n_tokens = self.param_groups[0]["n_tokens"]
        self._n_swaps = self.param_groups[0]["n_swaps"]
        self._local_swap_count = self.param_groups[0]["local_swap_count"]
        self._param = self.param_groups[0]["params"][0]
        self._negative_only = self.param_groups[0]["negative_only"]
        self._coord_randk = self.param_groups[0]["coord_randk"]
        self._global_topk = self.param_groups[0]["global_topk"]
        self._n_buffers = self.param_groups[0]["n_buffers"]
        self._ignore_curr_marginals = self.param_groups[0]["ignore_curr_marginals"]

    @property
    def coordinate(self) -> Coordinate:
        param = self._param
        if self._embedding:
            param = torch.matmul(param, self._embedding.weight.T)
        return Coordinate.from_tensor(param.argmax(-1))

    @property
    def coordinate_replacements(self) -> list[Coordinate]:
        if self._replacements is None:
            if self._param.grad is None:
                raise ValueError("Params must have gradients!")
            param = self._param
            if self._embedding:
                param = torch.matmul(param, self._embedding.weight.T)
                param.grad = torch.matmul(self._param.grad, self._embedding.weight.T)

            replacements = self._select_replacements(param)
            # _select_replacements is random so synchronize with all other processes
            self._replacements = broadcast_object_list(replacements, from_process=0)

        return self._replacements

    @coordinate_replacements.setter
    def coordinate_replacements(self, replacements: list[Coordinate] | None):
        self._replacements = replacements

    def _select_replacements(self, param) -> list[Coordinate]:
        # Explicitly ignore universally banned tokens by making gradients very large
        param.grad[..., self._ignored_values] = torch.inf

        # Explicitly ignore each of the current tokens and dictionary for all possible picks
        # FIXME: This is not right at all, it ignores everything?
        if self._ignore_curr_marginals:
            param.grad[torch.where(param)] = torch.inf

        assert (
            param.grad.ndim == 2
        ), "Need 'grad' to be 2D tensor of shape (n_tokens, n_dictionary)!"
        coords = torch.where(torch.isfinite(param.grad))
        coords = torch.stack(coords, dim=0).T

        # Select only negative dictionary gradients for each token position
        if self._negative_only:
            coords = torch.where(param.grad < 0)
            coords = torch.stack(coords, dim=0).T
            # Check if no tokens survive after self._negative_only
            if len(coords) == 0:
                # The entire optimization step will be skipped
                warnings.warn(
                    f"No tokens can be swapped at all! Check {self._negative_only = }!"
                )
                return []

        # Sample coord_randk random gradients for each token position separately
        if self._coord_randk > 0:
            coords = sample_coord_randk(coords, self._coord_randk)

        # Select coord_topk lowest gradients for each token position separately
        if self._coord_topk > 0:
            coords = pick_coord_topk(coords, param.grad, self._coord_topk)

        # Select global_topk lowest gradients with survival guarantees for all token positions
        if self._global_topk > 0:
            coords = pick_global_topk(coords, param.grad, self._global_topk)

        # Swap as many tokens as we can up to the requested tuple size
        _n_swap_tokens = self._n_tokens
        if len(torch.unique(coords[:, 0])) < self._n_tokens:
            _n_swap_tokens = len(torch.unique(coords[:, 0]))
            warnings.warn(
                f"Fewer than {self._n_tokens} token(s) survived, will instead swap tuples of {_n_swap_tokens} token(s)!"
            )

        # Sample _n_swap_tokens-tuples
        swaps = sample_swaps(coords, _n_swap_tokens, self._n_swaps)

        # Remove duplicate swaps
        swaps = pick_unique_swaps(swaps)

        # Verify we're not swapping the same token position twice in the same swap
        assert (
            torch.count_nonzero(swaps[:, :, None, 0] == swaps[:, None, :, 0], dim=2)
            == 1
        ).all(), "Attempting to swap the same token more than once in the same tuple!"

        return [Coordinate.from_tensor(swap) for swap in swaps]

    def update_parameter(self, coord: Coordinate):
        self._param.data[coord.index] = 0.0
        if self._embedding is None:
            self._param.data[coord.index, coord.value] = 1.0
        else:
            # Convert values to embeddings and use basic indexing to set embeds
            embeds = self._embedding(coord.to_tensor(self._embedding.device)[..., 1])
            for idx, embed in zip(coord.index, embeds):
                self._param.data[idx].copy_(embed)

    @torch.inference_mode()
    def step(self, closure: Callable):  # type: ignore
        # Stay up to date with the latest hyper-parameters
        self._update_hyperparams()

        # Lazily initialize state
        p_state = self.state[self._param]
        p_state["swap_count"] = p_state.get(
            "swap_count", torch.tensor(0, device=self._param.device)
        )
        p_state["buffers"] = p_state.get("buffers", [])

        # Split indexed replacements across devices and compute losses
        replacements = self.coordinate_replacements
        idxes = torch.arange(len(replacements), device=self._param.device)
        with PartialState().split_between_processes(
            list(zip(idxes, replacements))
        ) as local_replacements:
            local_outputs = self._step(closure, local_replacements)  # type: ignore
            local_outputs = pad_across_processes(local_outputs, pad_index=-1)  # type: ignore

        # Gather global indices and losses and select valid ones
        idxes, losses = gather(local_outputs)
        is_valid = torch.where((idxes != -1) & torch.isfinite(losses))[0]  # type: ignore
        idxes, losses = idxes[is_valid], losses[is_valid]

        if len(losses) == 0:
            warnings.warn(f"All {self._n_swaps} attempted swaps do not re-encode!")

        # Add losses to buffer popping max loss if we exceed length
        old_coord = self.coordinate
        for idx, loss in zip(idxes, losses):
            coord = old_coord | replacements[idx]
            bisect.insort(p_state["buffers"], (loss, coord), key=itemgetter(0))
            if len(p_state["buffers"]) > self._n_buffers:
                p_state["buffers"].pop()

        # Update parameter with smallest loss coordinate
        if len(p_state["buffers"]) > 0:
            _, best_coord = p_state["buffers"].pop(0)
            self.update_parameter(best_coord)

        # Save information about replaced tokens
        p_state["swap_count"] += reduce(self._local_swap_count, "sum")
        self._local_swap_count.zero_()

        # Force next call to step to update replacements
        self.coordinate_replacements = None

        # Safety check that param is still 1-hot or a direct embedding
        if self._embedding is None:
            assert (self._param.sum(dim=-1) == 1.0).all()
        else:
            assert torch.allclose(
                self._param,
                self._embedding(
                    torch.cdist(self._param, self._embedding.weight).argmin(-1)
                ),
            )

    def _step(
        self, closure: Callable, replacements: list[tuple[torch.Tensor, Coordinate]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generator = None
        if inspect.isgeneratorfunction(closure):
            generator = closure()
            generator.send(None)

        old_coord = self.coordinate

        idx_losses = []
        for idx, new_coord in tqdm(replacements, desc="replacements", leave=False):
            # Vector replace coordinates with new value
            self.update_parameter(new_coord)
            idx_losses.extend(generator.send(idx) if generator else [(idx, closure())])
            self.update_parameter(old_coord)
            self._local_swap_count += 1

        # Get remaining losses, if any
        if generator:
            idx_losses.extend(generator.send(None))
            generator.close()

        # Return dummy index and loss when empty otherwise stack them
        if len(idx_losses) == 0:
            return (
                torch.tensor([-1], device=self._param.device),
                torch.tensor([torch.inf], device=self._param.device),
            )
        idxes, losses = list(zip(*idx_losses))
        return torch.stack(idxes), torch.stack(losses)


def from_config(cfg: OptimConf, params: ParamsT, **gcg_kwargs) -> Optimizer:
    """Creates an optimizer instance from configuration.

    Args:
        cfg: Optimizer configuration object.
        params: Model parameters to optimize.
        **gcg_kwargs: Additional keyword arguments for GCG optimizer.

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: If optimizer name is unknown.
    """

    if cfg.name == "gcg":
        return GreedyCoordinateGradient(
            params,
            negative_only=cfg.negative_only,
            coord_randk=cfg.coord_randk,
            coord_topk=cfg.coord_topk,
            global_topk=cfg.global_topk,
            n_tokens=cfg.n_tokens,
            n_swaps=cfg.n_swaps,
            n_buffers=cfg.n_buffers,
            **gcg_kwargs,
        )

    if cfg.name == "sgd":
        return SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            dampening=cfg.dampening,
            weight_decay=cfg.weight_decay,
            nesterov=cfg.nesterov,
        )

    if cfg.name == "adam":
        return Adam(
            params,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            amsgrad=cfg.amsgrad,
        )

    raise ValueError(f"Unknown optimizer: {cfg.name}")
