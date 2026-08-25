#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import torch
from typing import cast
from transformers import BatchEncoding

from llmart import DataMapper
from .basic import BasicBuilder


class MllamaSampleBuilder(BasicBuilder):
    def _generate_examples(self, **kwargs):
        yield (
            0,
            {
                "image": "https://llava-vl.github.io/static/images/view.jpg",
                "question": "What does the image show?",
                "response": "That is a dog!",
            },
        )


class MllamaSampleMapper(DataMapper):
    def make_conversation(self, image, question, response):
        return [
            dict(
                role="user",
                content=[
                    dict(type="image", url=image),
                    dict(type="text", text=self.modify_prompt(question)),
                ],
            ),
            dict(
                role="assistant",
                content=[
                    dict(type="text", text=self.force_completion(response)),
                ],
            ),
        ]

    def __call__(self, batch):
        # Create conversation data structure and mark parts we care about
        convs = [
            self.make_conversation(*args)
            for args in zip(batch["image"], batch["question"], batch["response"])
        ]

        # Turn conversation into inputs_ids and masks
        inputs = cast(
            BatchEncoding,
            self.processor.apply_chat_template(
                convs,  # pyright: ignore[reportArgumentType]
                padding=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            ),
        )

        # Add batch axis to tensor values (e.g., pixel_values)
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and len(value) != 1:
                inputs[key] = value[None]

        # Construct labels from response_mask
        response_mask = inputs["response_mask"]
        input_ids = inputs["input_ids"]
        assert isinstance(response_mask, torch.Tensor)
        assert isinstance(input_ids, torch.Tensor)
        labels = input_ids.clone()
        labels[~response_mask] = -100
        inputs["labels"] = labels

        return inputs.data
