#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import torch

from .utils import get_survivors, early_exit_on_empty


@early_exit_on_empty
def sample_coord_randk(coords: torch.Tensor, coord_randk: int) -> torch.Tensor:
    """Samples coordinates using random-k selection for each unique coordinate.

    Args:
        coords: Tensor of shape (n_swaps, 2), where first column contains token indices
            and second column contains token ids.
        coord_randk: Maximum number of random samples to select per token coordinate.

    Returns:
        torch.Tensor: Concatenated tensor of sampled coordinates.

    Raises:
        AssertionError: If no valid candidates found for a token position.
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

        # Pick at most random-k
        picked_rows = valid_rows[
            torch.randperm(len(valid_rows), device=coords.device)[:coord_randk]
        ]
        output_coords.append(coords[picked_rows])

    # Concatenate outputs
    output_coords = torch.cat(output_coords, dim=0)
    return output_coords


def sample_swaps(coords: torch.Tensor, n_tokens: int, n_swaps: int) -> torch.Tensor:
    """Generates multiple sets of token position swaps from input coordinates.

    Args:
        coords: Tensor of shape (n_swaps, 2), where first column contains token indices
            and second column contains token ids.
        n_tokens: Number of token positions to sample for each swap.
        n_swaps: Number of different swap combinations to generate.

    Returns:
        torch.Tensor: Tensor containing sampled coordinate combinations.
        Shape: (n_swaps,  n_tokens, 2).
    """

    # Get the number of surviving token positions in the input
    valid_tokens, n_valid_tokens, _ = get_survivors(coords)

    # Create masks for each token position
    masks = [(coords[:, 0] == id) for id in valid_tokens]

    # Pre-sample max size tuples indexing the original tensor for all n_swaps samples
    max_tuples = [mask.nonzero().squeeze(1) for mask in masks]
    max_tuples = [
        samples[torch.randint(len(samples), (n_swaps,))] for samples in max_tuples
    ]
    max_tuples = torch.stack(max_tuples, dim=1)  # Shape: (n_swaps, n_valid_tokens)

    # Sample n_tokens positions for each sample
    sampled_positions = torch.argsort(
        torch.randint(low=0, high=n_valid_tokens, size=(n_swaps, n_valid_tokens)),
        dim=-1,
    )[:, :n_tokens]  # Shape: (n_swaps, n_tokens)

    # Reduce max size tuples to n_tokens-tuples via advanced indexing
    row_indices = torch.arange(n_swaps).unsqueeze(1).expand(-1, n_tokens)
    output_coords_idxs = max_tuples[row_indices, sampled_positions]

    # Gather
    output_coords = coords[output_coords_idxs]
    return output_coords
