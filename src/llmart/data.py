#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import torch
import itertools
from typing import Any
from collections import defaultdict
from collections.abc import Generator, MutableMapping
from accelerate.utils import gather, pad_across_processes
from importlib import import_module
from datasets import load_dataset, Dataset, DatasetDict

from .config import DataConf


def microbatch(
    inputs: MutableMapping[str, Any], micro_batch_size: int
) -> Generator[MutableMapping[str, Any], None, None]:
    """Splits input data into smaller batches.

    Args:
        inputs: Dictionary of input tensors to be split into micro-batches.
        micro_batch_size: Maximum size of each micro-batch.

    Returns:
        Generator yielding dictionaries of input tensors split into micro-batches.
    """

    total_samples = len(inputs["input_ids"])
    for start_idx in range(0, total_samples, micro_batch_size):
        end_idx = min(start_idx + micro_batch_size, total_samples)
        yield {k: v[start_idx:end_idx] for k, v in inputs.items()}


def gather_batch_across_processes(
    inputs: MutableMapping[str, Any],
    pad_index: int | dict[str, int] = 0,
    pad_first: bool | dict[str, bool] = True,
    dim: int | dict[str, int] = 1,
) -> MutableMapping[str, Any]:
    """Gathers and pads batched dictionary inputs across multiple processes.

    For each key in the inputs, pad_index and pad_first can be dictionaries with the same
    keys to specify per-key pad index values and whether to add padding at the beginning
    or end of the input.

    Args:
        inputs: Dictionary of input tensors to gather.
        pad_index: Padding value(s) for each input tensor.
        pad_first: Whether to pad at start (True) or end (False) of tensors.
        dim: Dimension along which to pad each tensor.

    Returns:
        Dictionary of gathered and padded input tensors.
    """

    pad_indices = (
        pad_index if isinstance(pad_index, dict) else defaultdict(lambda: pad_index)
    )
    pad_firsts = (
        pad_first if isinstance(pad_first, dict) else defaultdict(lambda: pad_first)
    )
    dims = dim if isinstance(dim, dict) else defaultdict(lambda: dim)

    # Pad all inputs. NOTE: we make all tensors the same dtype since pad_across_processes
    # is not safe and assumes default torch dtype.
    global_inputs = {
        k: pad_across_processes(
            inputs[k],
            dim=dims[k],
            pad_index=pad_indices[k],
            pad_first=pad_firsts[k],
        ).to(inputs[k].dtype)  # type: ignore
        for k in inputs.keys()
    }

    # Gather all inputs
    global_inputs = {k: gather(v) for k, v in global_inputs.items()}  # type: ignore

    return global_inputs


def from_config(
    cfg: DataConf,
    **kwargs,
) -> DatasetDict:
    """Creates dataset splits from configuration.

    Args:
        cfg: Configuration object containing dataset parameters.

    Returns:
        DatasetDict containing train, validation, test and mini-train splits.

    Raises:
        ValueError: If requested subset indices are out of bounds.
        NotImplementedError: If dataset has predefined val/test splits.
    """

    try:
        local_dataset = import_module(f".datasets.{cfg.path}", __package__)
        if local_dataset.__file__ is not None:
            cfg.path = local_dataset.__file__
    except ModuleNotFoundError:
        pass  # ignore issues importing local dataset and let load_dataset raise them

    dd = load_dataset(
        cfg.path,
        data_files=cfg.files,
        trust_remote_code=cfg.trust_remote_code,
        **kwargs,
    )
    if not isinstance(dd, DatasetDict):
        raise ValueError(f"Dataset must return a DatasetDict, got: {dd.__class__}")
    if "input_ids" not in dd["train"].features:
        raise ValueError(
            f"Training dataset must have input_ids, has: {dd['train'].features}"
        )

    # Subselect training samples before random splits
    if cfg.subset is not None:
        if max(cfg.subset) >= len(dd["train"]):
            raise ValueError(
                f"{cfg.subset=} is out of bounds for dataset with {len(dd['train'])} training samples"
            )
        dd["train"] = dd["train"].select(cfg.subset)

    train, minitrain, val, test = _split_dataset(
        train=dd["train"],
        val=dd.get("val", None),
        test=dd.get("test", None),
        n_train=cfg.n_train,
        n_minitrain=cfg.n_minitrain,
        n_val=cfg.n_val,
        n_test=cfg.n_test,
        shuffle=cfg.shuffle,
    )

    # Compress datasets using attention_mask
    mask_name = "attention_mask"
    if "response_mask" in train.features:
        mask_name = "response_mask"
    train = _compress(train, mask_name=mask_name)
    minitrain = _compress(minitrain, mask_name=mask_name)
    val = _compress(val, mask_name=mask_name)
    test = _compress(test, mask_name=mask_name)

    return DatasetDict(train=train, minitrain=minitrain, val=val, test=test)


def _split_dataset(
    train: Dataset,
    val: Dataset | None = None,
    test: Dataset | None = None,
    n_train: int | None = None,
    n_minitrain: int | None = None,
    n_val: int | None = None,
    n_test: int | None = None,
    shuffle: bool = False,
) -> tuple[Dataset, Dataset, Dataset, Dataset]:
    n_train = n_train if n_train is not None and n_train > 0 else None
    n_val = n_val if n_val is not None and n_val > 0 else None
    n_test = n_test if n_test is not None and n_test > 0 else None

    if n_test is None and n_val is None:
        # If unspecified test, then reuse entire train as test
        test = test or train
        n_test = n_test or len(test)
    elif n_test is None and test is None:
        # Protect against case when user specified val but no test
        raise AttributeError(
            f"n_test should be > 0, is {n_test} or test dataset should be specified"
        )

    # Split train into val and/or test
    if val is None and test is None:
        assert n_test is not None
        dd = train.train_test_split((n_val or 0) + n_test, n_train, shuffle)
        train = dd["train"]
        val = dd["test"].take(n_val or 0)
        test = dd["test"].skip(n_val or 0).take(n_test)

    elif val is None:
        dd = (
            DatasetDict(train=train, test=train.take(0))
            if n_val is None
            else train.train_test_split(n_val, n_train, shuffle)
        )
        train = dd["train"]
        val = dd["test"]

    elif test is None:
        dd = train.train_test_split(n_test, n_train, shuffle)
        train = dd["train"]
        test = dd["test"]
    assert test is not None

    # Take from datasets taking care to use entire dataset when unspecified and not
    # exceed the length of dataset
    train = train.take(min(n_train or len(train), len(train)))
    val = val.take(min(n_val or len(val), len(val)))
    test = test.take(min(n_test or len(test), len(test)))

    # Subsample from train, if specified defaulting to entire training set when None
    n_minitrain = len(train) if n_minitrain is None else n_minitrain
    minitrain = train.take(min(n_minitrain, len(train)))

    return train, minitrain, val, test


def _compress(ds: Dataset, mask_name: str = "attention_mask"):
    features = set(ds.features.keys())
    if mask_name not in features:
        return ds

    # Find mask that works will all examples
    mask = ds[mask_name]
    mask = torch.tensor(mask).any(0)  # find smallest mask
    mask = mask.flip((0,)).cumsum(0).flip((0,)) > 0  # left fill mask with ones
    mask = mask.to(torch.int).tolist()

    # Compress all features of example using mask
    def compress_example(data):
        for feat in features:
            data[feat] = list(itertools.compress(data[feat], mask))
        return data

    ds = ds.map(compress_example)
    return ds
