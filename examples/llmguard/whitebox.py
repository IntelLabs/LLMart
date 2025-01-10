#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import fire  # type: ignore[reportMissingImports]
import torch
from collections.abc import MutableMapping
from transformers import pipeline
from llmart import (
    AttackPrompt,
    AdversarialTextClassificationPipeline,
    GreedyCoordinateGradient,
)


def attack(
    text: str,
    *,
    label: str = "SAFE",
    target_score: float = 0.9,
    suffix_length: int = 2,
    suffix_init: str = "@",
    seed: int = 2024,
    model: str = "ProtectAI/deberta-v3-base-prompt-injection-v2",
    revision: str = "e6535ca4ce3ba852083e75ec585d7c8aeb4be4c5",
    device: str = "cuda",
    gcg_n_tokens: int = 1,
):
    torch.manual_seed(seed)

    classifier = pipeline(
        task="text-classification",
        model=model,
        revision=revision,
        truncation=True,
        max_length=512,
        device=device,
    )

    # We're running a suffix attack with 2 adversarial tokens
    adv_classifier = pipeline(
        task="adv-text-classification",
        model=classifier.model,
        tokenizer=classifier.tokenizer,
        attack=AttackPrompt(suffix=suffix_length, default_token=suffix_init),
        device=device,
    )

    assert isinstance(adv_classifier, AdversarialTextClassificationPipeline)
    optim = GreedyCoordinateGradient(
        adv_classifier.attack.parameters(),
        ignored_values=adv_classifier.tokenizer.bad_token_ids,
        n_tokens=gcg_n_tokens,
    )

    while True:

        def closure(return_outputs=False):
            outputs: MutableMapping = adv_classifier(text, label=label)[0]  # type: ignore
            return outputs if return_outputs else outputs["loss"]

        outputs = closure(return_outputs=True)
        loss = outputs["loss"]

        # Check if we found an attack that works
        with torch.inference_mode():
            adv_text = outputs["text"]
            out: MutableMapping = classifier(adv_text)[0]  # type: ignore
            print(f"{out} {adv_text}")
            if out["label"] == label and out["score"] > target_score:
                break

        # If not, then run an iteration of the attack
        loss.backward()
        optim.step(closure)

    print(adv_text, "=>", classifier(adv_text))


if __name__ == "__main__":
    fire.Fire(attack)
