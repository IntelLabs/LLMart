#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import nltk
nltk.download('punkt_tab')
import fire  # type: ignore[reportMissingImports]
import torch
import os
import re
from collections import defaultdict

from minicheck.minicheck import MiniCheck  # type: ignore[reportMissingImports]
from minicheck.utils import SYSTEM_PROMPT, USER_PROMPT  # type: ignore[reportMissingImports]
from transformers import pipeline, AutoTokenizer

from llmart import (
    AdversarialTextGenerationPipeline,
    GreedyCoordinateGradient,
    AttackPrompt,
)
from llmart.pipelines.text_generation import Chat


def attack(suffix=6, n_swaps=1024, n_tokens=1, num_steps=1000, per_device_bs=64):
    claim = "Platypuses have venomous spurs on feet."
    doc = "The old oak tree stood tall, its gnarled branches reaching towards the sky like ancient fingers. Beneath its sprawling canopy, generations of children had played, lovers had met, and weary travelers had found respite from the midday sun."

    # Get reference pipeline
    scorer = MiniCheck(
        model_name="Bespoke-MiniCheck-7B",
        enable_prefix_caching=False,
        cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
    )

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "bespokelabs/Bespoke-MiniCheck-7B", trust_remote_code=True
    )

    # Get HF pipeline
    pipe = pipeline(
        "text-generation",
        model="bespokelabs/Bespoke-MiniCheck-7B",
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map="cuda:1",
        torch_dtype="bfloat16",
        max_new_tokens=1,
    )

    # Attack pipeline
    adv_pipe = pipeline(
        "adv-text-generation",
        model=pipe.model.requires_grad_(False),
        tokenizer=pipe.tokenizer,
        attack=AttackPrompt(suffix=suffix),
    )
    assert isinstance(adv_pipe, AdversarialTextGenerationPipeline)

    # Attack optimizer
    optimizer = GreedyCoordinateGradient(
        adv_pipe.attack.parameters(),
        ignored_values=adv_pipe.tokenizer.bad_token_ids,  # Consider only ASCII solutions
        coord_randk=2**14,
        global_topk=0,
        n_tokens=n_tokens,
        n_swaps=n_swaps,
    )

    # Get the "yes" dictionary entries
    YES_TOKEN_IDS = [
        idx for token, idx in tokenizer.vocab.items() if token.lower() == "yes"
    ]

    conversation = [
        dict(role="system", content=SYSTEM_PROMPT),
        dict(
            role="user",
            content=USER_PROMPT.replace("[DOCUMENT]", doc).replace("[CLAIM]", claim),
        ),
        dict(role="assistant", content=tokenizer.decode([YES_TOKEN_IDS[0]])),
    ]

    def generator():
        param_losses = []
        batch = defaultdict(list)

        while True:
            # Get next attack
            param_idx = yield param_losses

            # If we have a whole batch, or a partial batch and we're stopping,
            # then compute per-example losses
            if (len(batch["param_idx"]) == per_device_bs) or (
                param_idx is None and len(batch["param_idx"]) > 0
            ):
                param_idxs = batch.pop("param_idx")
                # Turn preprocessed inputs into model inputs for Pipeline's forward
                model_inputs = {
                    key: torch.cat(value)
                    if isinstance(value[0], torch.Tensor)
                    else value
                    for key, value in batch.items()
                }
                outputs = adv_pipe.forward(
                    model_inputs,
                    inf_loss_when_nonreencoding=True,
                    use_cache=False,
                )

                # FIXME: Not sure why the type isn't correct when dereferencing outputs
                logits = outputs["logits"][..., -2, :]  # type: ignore

                # Find "yes" logits and select the one with lowest NLL to minmiize
                all_dict_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                target_logprobs = all_dict_logprobs[..., YES_TOKEN_IDS]

                # Compute per-example loss but use model loss when infinite
                model_losses = outputs["losses"]  # type: ignore
                losses = torch.min(-target_logprobs, dim=-1)[0]
                losses = torch.where(model_losses == torch.inf, torch.inf, losses)
                param_losses = zip(param_idxs, losses)

                # Next iteration will start reaccumulating a batch
                batch = defaultdict(list)
            else:
                param_losses = []

            # If we're stopping, then yield any remaing losses
            if param_idx is None:
                yield param_losses
                break

            # Otherwise, attack inputs and make sure they reencode..
            adv_inputs = adv_pipe.preprocess(Chat(conversation))  # type: ignore
            if not adv_inputs["reencodes"].all():  # type: ignore
                continue

            # ...and if they do accumulate a batch
            batch["param_idx"].append(param_idx)
            for key, value in adv_inputs.items():
                batch[key].append(value)

    # Define a closure with custom loss function
    def closure(return_outputs=False) -> torch.Tensor | dict:
        outputs: dict = adv_pipe(
            conversation,
            inf_loss_when_nonreencoding=not return_outputs,
            use_cache=False,
        )[0]  # type: ignore
        outputs["model_loss"] = outputs["loss"]

        # Compute custom loss by inducing a specific "yes" tokens on the predicted output logit
        # This constants -2 is derived from the fact that max_new_tokens=1 and the model
        # predicts the next token.
        logits = outputs["logits"][..., -2, :]

        # Find "yes" logits and select the one with lowest NLL to minmiize
        all_dict_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        target_logprobs = all_dict_logprobs[..., YES_TOKEN_IDS]
        loss = torch.min(-target_logprobs)

        # inf model loss means this input does not reencode!
        if outputs["model_loss"] != torch.inf:
            outputs["loss"] = loss

        if return_outputs:
            return outputs
        return outputs["loss"]

    # For each step
    for step_idx in range(num_steps):
        outputs: dict = closure(return_outputs=True)  # type: ignore
        model_loss: torch.Tensor = outputs["model_loss"]
        loss: torch.Tensor = outputs["loss"]

        # Check claim
        with torch.no_grad():
            # Get adversarial user content
            adv_user_content = outputs["generated_text"][1]["content"]

            # Extract claim
            pattern = r"\nClaim: (.*)"
            adv_claim = re.search(pattern, adv_user_content, re.DOTALL).group(1).strip()  # type: ignore

            # Use reference pipeline for validation
            # NOTE: HF InternLM2 does many float casts but vLLM's InternLM2 uses blfoat16 throughout
            _, adv_prob, _, _ = scorer.score(docs=[doc], claims=[adv_claim])
            adv_prob = adv_prob[0]  # type: ignore
            print(
                f"{step_idx = }, {model_loss = :0.4f}, {loss = :0.4f}, {adv_prob = :0.4f}, {adv_claim = }"
            )

        # Back-propagate
        optimizer.zero_grad()
        loss.backward()

        # Optimizer
        with torch.no_grad():
            _new_loss = optimizer.step(generator)


if __name__ == "__main__":
    fire.Fire(attack)
