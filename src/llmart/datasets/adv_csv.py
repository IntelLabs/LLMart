#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import abc
from datasets import DatasetDict, Dataset
from datasets.packaged_modules import csv

from llmart import TaggedTokenizer, Transform, ConversationMapper


class AdvCsvConfig(csv.CsvConfig):
    def __init__(
        self,
        *args,
        tokenizer: TaggedTokenizer,
        mark_prompt: Transform,
        mark_completion: Transform,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.tokenizer = tokenizer
        self.mark_prompt = mark_prompt
        self.mark_completion = mark_completion


class AdvCsv(csv.Csv):
    BUILDER_CONFIG_CLASS = AdvCsvConfig

    @abc.abstractmethod
    def to_conversations(self, ds) -> list[list[dict]]:
        raise NotImplementedError

    def as_dataset(self, *args, **kwargs):
        tokenizer: TaggedTokenizer = self.config.tokenizer  # type: ignore

        # Get training dataset
        dd = super().as_dataset(*args, **kwargs)
        ds = dd["train"]

        # Convert dataset to conversations
        convs = self.to_conversations(ds)

        # Turn conversation into input_ids, labels, masks, etc.
        conv_mapper = ConversationMapper(tokenizer)
        inputs = conv_mapper(convs)
        ds = Dataset.from_dict(inputs)
        return DatasetDict(train=ds)
