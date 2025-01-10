#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import logging
from typing import Any
from collections import defaultdict
from collections.abc import Generator, MutableMapping
from accelerate.utils import gather, pad_across_processes
from importlib import import_module
from datasets import load_dataset as load_hf_dataset, DatasetDict, Dataset

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


def load_dataset(path: str, **kwargs) -> DatasetDict:
    """Loads a dataset from local module or Hugging Face.

    Args:
        path: Dataset path or identifier.
        **kwargs: Additional arguments passed to dataset loading function.

    Returns:
        Loaded dataset as a DatasetDict.
    """

    try:
        local_dataset = import_module(f".datasets.{path}", __package__)
        conversations = local_dataset.get_conversations(**kwargs)
        return DatasetDict(train=Dataset.from_dict({"conversation": conversations}))

    except ModuleNotFoundError:
        ds = load_hf_dataset(path, **kwargs)
        assert isinstance(ds, DatasetDict)
        return ds


def from_config(cfg: DataConf) -> DatasetDict:
    """Creates dataset splits from configuration.

    Args:
        cfg: Configuration object containing dataset parameters.

    Returns:
        DatasetDict containing train, validation, test and mini-train splits.

    Raises:
        ValueError: If requested subset indices are out of bounds.
        NotImplementedError: If dataset has predefined val/test splits.
    """

    kwargs = {}
    if cfg.files is not None:
        kwargs["data_files"] = cfg.files
    dd = load_dataset(cfg.path, **kwargs)

    train_ds = dd["train"]
    val_ds = dd.get("val", None)
    test_ds = dd.get("test", None)

    # Subselect training samples before random splits
    if cfg.subset is not None:
        if max(cfg.subset) >= len(dd["train"]):
            raise ValueError(
                f"{cfg.subset=} is out of bounds for dataset with {len(dd['train'])} training samples"
            )
        train_ds = dd["train"].select(cfg.subset)

    if val_ds is None and test_ds is None:
        # No val/test specified, so sample them from train
        if (cfg.n_train or 0) + cfg.n_val + cfg.n_test > len(train_ds):
            val_ds = train_ds.take(cfg.n_val)
            test_ds = train_ds.take(cfg.n_test)

        else:
            train_dd = train_ds.train_test_split(
                train_size=cfg.n_train,
                test_size=cfg.n_val + cfg.n_test,
                shuffle=cfg.shuffle,
            )
            train_ds = train_dd["train"]
            val_dd = train_dd["test"].train_test_split(
                train_size=cfg.n_val,
                test_size=cfg.n_test,
                shuffle=cfg.shuffle,
            )
            val_ds = val_dd["train"]
            test_ds = val_dd["test"]

    else:
        raise NotImplementedError

    # Subsample train
    minitrain_ds = train_ds.take(min(cfg.n_minitrain, len(train_ds)))
    if cfg.n_minitrain > len(train_ds):
        logging.warning(
            f"{cfg.n_minitrain=} larger than all training data; will take all of it instead"
        )

    return DatasetDict(train=train_ds, minitrain=minitrain_ds, val=val_ds, test=test_ds)
