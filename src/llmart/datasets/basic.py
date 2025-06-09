#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import torch
import datasets
from transformers.tokenization_utils_base import BatchEncoding

from llmart import DataMapper


class BasicBuilder(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo()

    def _split_generators(self, dl_manager):
        return [datasets.SplitGenerator(name="train")]

    def _generate_examples(self, **kwargs):
        example = dict(
            prompt="Tell me about the planet Saturn.", completion="NO WAY JOSE"
        )
        yield 0, example


class BasicMapper(DataMapper):
    def __call__(self, batch):
        # Mark conversation noting that they are batched
        convs = [
            [
                dict(role="user", content=self.modify_prompt(prompt)),
                dict(role="assistant", content=self.force_completion(completion)),
            ]
            for prompt, completion in zip(batch["prompt"], batch["completion"])
        ]

        # Turn conversation into input_ids and masks.
        # NOTE: One could use llmart.ConversationMapper or return bare conversations.
        inputs = self.tokenizer.apply_chat_template(
            convs,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )
        assert isinstance(inputs, BatchEncoding)

        # Construct labels from response_mask
        input_ids = inputs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        labels = input_ids.detach().clone()

        response_mask = inputs["response_mask"]
        assert isinstance(response_mask, torch.Tensor)
        labels[~response_mask] = -100

        inputs["labels"] = labels
        return inputs.data
