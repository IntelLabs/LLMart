#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import torch

from llmart.utils import get_survivors, early_exit_on_empty, setdiff1d
from llmart.utils import reorder_tensor


@early_exit_on_empty
def pick_coord_topk(
    coords: torch.Tensor, grad: torch.Tensor, coord_topk: int
) -> torch.Tensor:
    """Selects top-k coordinates for each token position based on gradients.

    Args:
        coords: Tensor of shape [N, 2] containing (token_position, candidate) pairs.
        grad: Tensor containing gradient values for each coordinate.
        coord_topk: Maximum number of coordinates to select per token position.

    Returns:
        Tensor of shape [M, 2] containing selected coordinates, where M ≤ N.
        Each row contains (token_position, candidate) pairs for the selected
        coordinates.

    Raises:
        AssertionError: If no candidates are found for a token position.
    """

    # Get the number of surviving token positions in the input
    valid_tokens, _, _ = get_survivors(coords)

    # For each unique coordinate
    output_coords = []
    for desired_coord in valid_tokens:
        valid_rows = torch.where(coords[:, 0] == desired_coord)[0]
        assert (
            len(valid_rows) > 0
        ), f"No candidates found for swapping token position {desired_coord}!"

        # Get the gradients too, we need them to sort
        # !!! Uses advanced indexing
        # [K, 2] -> transpose to [2, K] -> split for advanced indexing
        valid_grads = grad[coords[valid_rows].T[0], coords[valid_rows].T[1]]

        # Sort the coordinates by values
        sorted_valid_rows = valid_rows[torch.argsort(valid_grads, descending=False)]

        # Pick at most top-k
        picked_rows = sorted_valid_rows[:coord_topk]
        output_coords.append(coords[picked_rows])

    # Concatenate outputs
    output_coords = torch.cat(output_coords, dim=0)
    return output_coords


@early_exit_on_empty
def pick_global_topk(
    coords: torch.Tensor, grads: torch.Tensor, global_topk: int
) -> torch.Tensor:
    """Selects top-k coordinates globally while ensuring coverage of all token positions.

    Args:
        coords: Tensor of shape [N, 2] containing (token_position, candidate) pairs.
        grads: Tensor containing gradient values for each coordinate.
        global_topk: Total number of coordinates to select across all positions.

    Returns:
        Tensor of shape [K, 2] containing selected coordinates, where K = global_topk.
        Each row contains (token_position, candidate) pairs for the selected
        coordinates.
    """

    # Get the number of surviving token positions in the input
    valid_tokens, n_valid_tokens, _ = get_survivors(coords)

    # Get all valid gradients and sort them
    valid_grads = grads[coords[:, 0], coords[:, 1]]
    sorted_valid_grads_idx = torch.argsort(valid_grads, descending=False)

    # We can only safely pick first (global_topk - n_valid_tokens + 1) winners
    # because all global_topk grads could be in only 1 coordinate
    num_safe_winners = global_topk - n_valid_tokens + 1
    output_coords = coords[sorted_valid_grads_idx[:num_safe_winners]]

    # Check if the winners so far cover all token positions
    covered_idxs = torch.unique(output_coords[:, 0])

    if len(covered_idxs) == n_valid_tokens:
        # Pick remaining coordinates
        output_coords = coords[sorted_valid_grads_idx[:global_topk]]

    else:
        # Find the unpicked token positions
        missing_idxs = setdiff1d(n_valid_tokens, covered_idxs)
        # Allocate to each a fair split of the remaining picks
        picks_per_remaining_tokens = (global_topk - num_safe_winners) // len(
            missing_idxs
        )
        assert (
            picks_per_remaining_tokens > 0
        ), "Some token swaps are completely dropped!"

        # For each unpicked token position, pick its top-most survivors
        extra_coords = []
        for missing_idx in missing_idxs:
            # Find coordinates for this token position
            token_coords = coords[coords[:, 0] == valid_tokens[missing_idx]]

            # Sort gradients for this token position
            token_grads = grads[token_coords[:, 0], token_coords[:, 1]]
            token_sorted_idx = torch.argsort(token_grads, descending=False)

            # Pick top survivors for this token position
            top_token_coords = token_coords[
                token_sorted_idx[:picks_per_remaining_tokens]
            ]
            extra_coords.append(top_token_coords)

        output_coords = torch.cat(
            [output_coords, torch.cat(extra_coords, dim=0)], dim=0
        )

    return output_coords


@early_exit_on_empty
def pick_unique_swaps(swaps: torch.Tensor) -> torch.Tensor:
    """Remove duplicate swaps.

    It is possible that a samplers can sample the same swaps if there is a small
    enough search space.

    Args:
        swaps: Tensor of shape (n_swaps, n_tokens, 2)

    Returns:
        torch.Tensor: Tensor of same shape but containing only unique swaps
    """
    # Sort based on swapped token index
    sorting_idx = swaps[..., 0].argsort(dim=-1)

    # Sort swaps
    sorted_swaps = reorder_tensor(swaps, sorting_idx)

    # Get unique swaps
    unique_swaps = torch.unique(sorted_swaps, sorted=False, dim=0)
    return unique_swaps
