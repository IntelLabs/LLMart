#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import torch
import inspect
import itertools
from typing import Any
from abc import ABC, abstractmethod
from collections import defaultdict
from importlib import import_module
from collections.abc import Generator, MutableMapping
from accelerate.utils import gather, pad_across_processes
from datasets import load_dataset, Dataset, DatasetDict
from datasets.load import LocalDatasetModuleFactoryWithScript
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding

from .tokenizer import TaggedTokenizer
from .transforms import Transform
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


class DataMapper(ABC):
    """
    Abstract base class for mapping raw dataset batches into model-ready input formats.

    This class defines the interface for transforming batches of raw data (e.g., from a HuggingFace Dataset)
    into tokenized and processed inputs suitable for model training or inference. Subclasses should implement
    the __call__ method to perform the mapping, which typically includes tokenization, prompt modification,
    and label construction.

    Args:
        tokenizer (PreTrainedTokenizerBase): Tokenizer used to encode text data.
        processor (ProcessorMixin): Optional processor for additional preprocessing (e.g., images).
        modify_prompt (Transform): Transformation applied to prompts before tokenization.
        force_completion (Transform): Transformation to enforce completion formatting or constraints.

    Methods:
        __call__(batch: dict[str, Any]) -> dict[str, Any]:
            Abstract method to map a batch of raw data to model inputs.
        load_data_mapper(path, trust_remote_code=False, **kwargs) -> "DataMapper":
            Loads a DataMapper subclass from a given path, supporting dynamic import and remote code if needed.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        processor: ProcessorMixin,
        modify_prompt: Transform,
        force_completion: Transform,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.modify_prompt = modify_prompt
        self.force_completion = force_completion

    @abstractmethod
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def load_data_mapper(
        path, trust_remote_code: bool = False, **kwargs
    ) -> "DataMapper":
        # Re-use dataset module factory to load mapper
        mapper_module = LocalDatasetModuleFactoryWithScript(
            path, trust_remote_code=trust_remote_code
        )
        module = import_module(mapper_module.get_module().module_path)
        for cls in module.__dict__.values():
            # Find classes that subclass DataMapper, are not abstract, and are in the path
            if (
                inspect.isclass(cls)
                and issubclass(cls, DataMapper)
                and not inspect.isabstract(cls)
                and inspect.getmodule(cls) == module
            ):
                return cls(**kwargs)
        raise ValueError(f"Unable to find DataMapper in {path}!")


class ConversationMapper(DataMapper):
    def __call__(self, batch):
        convs = batch["conversation"]

        # Tokenize a conversation
        inputs = self.tokenizer.apply_chat_template(
            convs,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )
        assert isinstance(inputs, BatchEncoding)

        # Check if inputs reencode
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        if (
            isinstance(self.tokenizer, TaggedTokenizer)
            and not self.tokenizer.reencodes(input_ids, attention_mask).all()
        ):
            raise ValueError(
                "There is some set of tokens in the conversation that do not re-encode."
            )

        # Construct labels from response mask, by setting non-response token labels to
        # the ignore_index value (which is -100 by default)
        response_mask = inputs.get("response_mask", None)
        if response_mask is not None:
            assert isinstance(response_mask, torch.Tensor)
            labels = input_ids.clone()
            labels[~response_mask] = -100
            inputs["labels"] = labels

        return inputs.data


def from_config(
    cfg: DataConf,
    **mapper_kwargs,
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

    # Load dataset and ensure we have a DatasetDict
    dd = load_dataset(
        cfg.path,
        name=cfg.name,
        split=cfg.split,
        data_files=cfg.files,
        trust_remote_code=cfg.trust_remote_code,
    )
    if isinstance(dd, Dataset):
        dd = DatasetDict(train=dd)
    if not isinstance(dd, DatasetDict):
        raise ValueError(f"Dataset must return a DatasetDict, got: {dd.__class__}")
    if "train" not in dd:
        raise ValueError(f"Dataset must have a train split, got: {list(dd.keys())}")

    # Map dataset into input_ids, attention_mask, etc.
    if cfg.mapper is not None:
        try:
            local_dataset = import_module(f".datasets.{cfg.mapper}", __package__)
            if local_dataset.__file__ is not None:
                cfg.mapper = local_dataset.__file__
        except ModuleNotFoundError:
            pass  # ignore issues importing local mapper and let load_data_mapper raise them

        mapper = DataMapper.load_data_mapper(
            cfg.mapper,
            **mapper_kwargs,
            trust_remote_code=cfg.trust_remote_code,
        )
        dd = dd.map(
            mapper,
            batched=True,
            remove_columns=dd.column_names["train"],
            load_from_cache_file=False,
        )
    if "conversation" in dd["train"].features:
        mapper = ConversationMapper(**mapper_kwargs)
        dd = dd.map(
            mapper,
            batched=True,
            remove_columns=dd.column_names["train"],
            load_from_cache_file=False,
        )
    if "input_ids" not in dd["train"].features:
        raise ValueError(
            f"Training dataset must have input_ids, has: {dd['train'].features}"
        )

    # Subselect training samples before splitting into train, val ,test
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

    # Compress datasets using attention_mask and, optionally, response_mask
    train = _compress(train)
    minitrain = _compress(minitrain)
    val = _compress(val)
    test = _compress(test)
    if "response_mask" in train.features:
        # Left fill since response_mask only masks the response and we want to keep
        # everything to the left of the response too.
        train = _compress(train, mask_name="response_mask", fill_side="left")
        minitrain = _compress(minitrain, mask_name="response_mask", fill_side="left")
        val = _compress(val, mask_name="response_mask", fill_side="left")
        test = _compress(test, mask_name="response_mask", fill_side="left")

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


def _compress(
    ds: Dataset, mask_name: str = "attention_mask", fill_side: str | None = None
):
    """
    Compresses the dataset by applying a mask to its features.

    Args:
        ds (Dataset): The dataset to compress.
        mask_name (str): The name of the mask feature in the dataset. Defaults to "attention_mask".
        fill_side (str | None): Determines how the mask is applied. Valid values are:
            - None: No special handling; the mask is applied as-is.
            - "left": The mask is flipped, cumulatively summed, and flipped back to fill from the left.
              This ensures that the left side of the mask is filled with ones.

    Returns:
        Dataset: The compressed dataset with features masked accordingly.

    Raises:
        NotImplementedError: If `fill_side` is not None or "left".
    """
    if fill_side not in (None, "left"):
        raise NotImplementedError(
            "Only fill_side=None and fill_side=left are currently supported!"
        )

    features = set(ds.features.keys())
    if mask_name not in features:
        return ds

    # Find mask that works will all examples
    mask = ds[mask_name]
    mask = torch.tensor(mask).any(0)  # find smallest mask
    if fill_side == "left":
        mask = mask.flip((0,)).cumsum(0).flip((0,)) > 0  # left fill mask with ones
    mask = mask.to(torch.int).tolist()

    # Compress all features of example using mask
    def compress_example(data):
        for feat in features:
            # Ignore features, like pixel_values and image_grid_thw, that don't match the mask size
            if len(mask) != len(data[feat]):
                continue
            data[feat] = list(itertools.compress(data[feat], mask))
        return data

    ds = ds.map(compress_example)
    return ds
