#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import torch
import functools


def setdiff1d(num_coords: int, candidate_set: torch.Tensor) -> torch.Tensor:
    """Computes set difference between range(num_coords) and candidate_set.

    A PyTorch implementation of numpy's setdiff1d for bounded integers, optimized
    for GPU usage.

    Args:
        num_coords: The upper bound of the range to compare against.
        candidate_set: Tensor containing the elements to subtract from range.

    Returns:
        Tensor containing elements in range(num_coords) that are not in candidate_set.
    """

    combined = torch.cat(
        (torch.arange(num_coords, device=candidate_set.device), candidate_set)
    )
    uniques, counts = combined.unique(return_counts=True)
    missing = uniques[counts == 1]  # Only appears once in the guaranteed arange

    return missing


def get_survivors(coords: torch.Tensor) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Analyzes token positions to find surviving unique tokens.

    Args:
        coords: Tensor containing token coordinates where first column is token indices.

    Returns:
        A tuple containing:
            - Tensor of unique valid token indices
            - Number of valid tokens
            - Tensor of counts for each valid token
    """

    # Get the number of surviving token positions in the input
    valid_tokens, counts = coords[:, 0].unique(return_counts=True)
    n_valid_tokens = len(counts)

    return valid_tokens, n_valid_tokens, counts


def reorder_tensor(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Reorders a tensor along its second dimension according to provided indices.

    Args:
        data: Input tensor of shape (M, K, P)
        indices: Integer tensor of shape (M, K) containing permutation indices

    Returns:
        Reordered tensor of shape (M, K, P)
    """
    M, K = data.shape[:2]

    # Create a tensor of indices for all dimensions
    batch_idx = torch.arange(M, device=data.device)[:, None].expand(M, K)

    # Stack the indices to create a (M, K, P) index tensor
    idx = torch.stack([batch_idx, indices], dim=-1)

    # Perform the reordering
    reordered = data[idx[..., 0], idx[..., 1]]

    return reordered


def early_exit_on_empty(func):
    """Decorator that returns empty tensor if input tensor is empty.

    Args:
        func: The function to decorate.

    Returns:
        Decorated function that checks for empty input tensor and returns early
        if found, otherwise executes the original function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Assume the first argument is the tensor
        if args and isinstance(args[0], torch.Tensor):
            tensor = args[0]
            if tensor.shape[0] == 0:
                return tensor
        return func(*args, **kwargs)

    return wrapper
