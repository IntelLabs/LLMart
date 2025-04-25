#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import torch
import pytest

from transformers import pipeline
from llmart import AttackPrompt


@pytest.fixture
def pipe():
    pipe = pipeline(
        "adv-text-generation",
        model="hf-internal-testing/tiny-gpt2-with-chatml-template",
        attack=AttackPrompt(suffix=20),
        model_kwargs=dict(),
    )

    return pipe


def test_adv_text_gen_prompt(pipe):
    outputs = pipe("This is a prompt", completion="This is a completion")
    assert isinstance(outputs, list)
    assert len(outputs) == 1

    outputs = outputs[0]
    assert (
        outputs["generated_text"]
        == "This is a prompt!!!!!!!!!!!!!!!!!!!! stairs stairs factors stairs"
    )
    assert (
        pipe.tokenizer.decode(outputs["labels"][outputs["labels"] >= 0])
        == "This is a completion"
    )
    assert outputs["loss"] == torch.inf


def test_adv_text_gen_prompt_without_completion(pipe):
    with pytest.raises(ValueError):
        pipe("This is a prompt without a completion")


def test_adv_text_gen_inf_loss(pipe):
    outputs = pipe(
        "This is a prompt",
        completion="This is a completion",
        inf_loss_when_nonreencoding=False,
    )
    assert outputs[0]["loss"] != torch.inf


def test_adv_text_gen_chat(pipe):
    outputs = pipe(
        [
            dict(role="user", content="This is a user chat"),
            dict(role="assistant", content="This is an assistant chat"),
        ]
    )

    assert isinstance(outputs, list)
    assert len(outputs) == 1

    outputs = outputs[0]
    assert outputs["generated_text"] == [
        {"role": "user", "content": "This is a user chat!!!!!!!!!!!!!!!!!!!!"},
        {"role": "assistant", "content": " factors factors stairs stairs stairs"},
    ]
    assert (
        pipe.tokenizer.decode(outputs["labels"][outputs["labels"] >= 0])
        == "This is an assistant chat"
    )
