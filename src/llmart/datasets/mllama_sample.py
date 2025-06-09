#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import torch

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
        inputs = self.processor.apply_chat_template(  # type: ignore[reportCallIssue]
            convs, padding=True, return_tensors="pt", return_dict=True, tokenize=True
        )

        # Add batch axis to tensor values (e.g., pixel_values)
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor) and len(inputs[key]) != 1:
                inputs[key] = inputs[key][None]

        # Construct labels from response_mask
        response_mask = inputs["response_mask"]
        inputs["labels"] = inputs["input_ids"].clone()
        inputs["labels"][~response_mask] = -100

        return inputs.data
