#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import datasets
from transformers import BatchEncoding

from llmart import TaggedTokenizer, Transform


class BasicConfig(datasets.BuilderConfig):
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


class BasicBuilder(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = BasicConfig

    def _info(self):
        return datasets.DatasetInfo()

    def _split_generators(self, dl_manager):
        del dl_manager
        return [datasets.SplitGenerator(name="train")]

    def _generate_examples(self, **kwargs):
        mark_prompt: Transform = self.config.mark_prompt  # type: ignore
        mark_completion: Transform = self.config.mark_completion  # type: ignore

        # Create conversation data structure and mark parts we care about
        conv = [
            dict(role="user", content=mark_prompt("Tell me about the planet Saturn.")),
            dict(role="assistant", content=mark_completion("NO WAY JOSE")),
        ]

        # Turn conversation into input_ids and masks
        inputs: BatchEncoding = self.config.tokenizer.apply_chat_template(  # type: ignore
            conv, return_tensors="pt", return_dict=True
        )

        # Construct labels from response_mask
        response_mask = inputs["response_mask"]  # type: ignore
        inputs["labels"] = inputs["input_ids"].clone()  # type: ignore
        inputs["labels"][~response_mask] = -100  # type: ignore

        # Remove batch axis which apply_chat_template adds
        yield 0, {k: v[0] for k, v in inputs.items()}
