#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
# type: ignore

import fire  # type: ignore[reportMissingImports]
import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmart import (
    TaggedTokenizer,
    GreedyCoordinateGradient,
    AttackPrompt,
    MaskCompletion,
    AdversarialAttack,
)
from llmart.attack import make_closure
from llmart.losses import CausalLMLoss
from model_position import AdversarialBlockShift

torch.manual_seed(2025)


def attack(
    suffix=10,
    n_swaps=512,
    n_tokens=2,
    num_steps=500,
    per_device_bs=64,
    position_cadence=5,
):
    user_input = "In which nightly PyTorch version was self-attention first introduced, and when was it merged in the stable release?"
    induced_response = "Self-attention is not supported in PyTorch.<|eot_id|>"

    # Get HF model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        revision="5f0b02c75b57c5855da9ae460ce51323ea669d8a",
        device_map="auto",
        torch_dtype="bfloat16",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        revision="5f0b02c75b57c5855da9ae460ce51323ea669d8a",
    )
    tokenizer.clean_up_tokenization_spaces = False
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Create transforms that inject markers into prompts
    add_attack = AttackPrompt(suffix=suffix)
    force_response = MaskCompletion(replace_with=induced_response)

    # Make tokenizer aware of attack marker tokens
    wrapped_tokenizer = TaggedTokenizer(
        tokenizer, tags=add_attack.tags + force_response.tags
    )

    # Create and tokenize conversation
    conversation = [
        dict(role="user", content=add_attack(user_input)),
        dict(role="assistant", content=force_response("")),
    ]
    inputs = wrapped_tokenizer.apply_chat_template(
        conversation,
        return_tensors="pt",
        return_dict=True,
        continue_final_message=False,
    ).to(model.device)
    # Construct labels for loss function from response_mask
    response_mask = inputs["response_mask"]
    inputs["labels"] = inputs["input_ids"].clone()
    inputs["labels"][~response_mask] = -100

    # Get the adversarial token and block shift attacks
    token_attack = AdversarialAttack(inputs, model.get_input_embeddings()).to(
        model.device
    )
    position_attack = AdversarialBlockShift(
        inputs,
        embedding=model.get_input_embeddings(),
    ).to(model.device)

    # Apply the two attacks simultaneously
    attack = torch.nn.Sequential(token_attack, position_attack)

    # Optimizers
    optimizer_tokens = GreedyCoordinateGradient(
        token_attack.parameters(),
        ignored_values=wrapped_tokenizer.bad_token_ids,  # Consider only ASCII solutions
        n_tokens=n_tokens,
        n_swaps=n_swaps,
    )
    optimizer_position = GreedyCoordinateGradient(
        position_attack.parameters(),
        n_tokens=1,
    )

    # Get closure to pass to discrete optimizer
    closure, closure_inputs = make_closure(
        attack,
        model,
        loss_fn=CausalLMLoss(),
        is_valid_input=wrapped_tokenizer.reencodes,
        batch_size=per_device_bs,
        use_kv_cache=False,  # NOTE: KV caching is incompatible with optimizable position
        ignored_keys=wrapped_tokenizer.mask_names
        + ["inputs_embeds"],  # NOTE: AdversarialBlockShift returns inputs_embeds
    )

    # For each step
    for step in (pbar := tqdm(range(num_steps))):
        optimizer_tokens.zero_grad()
        optimizer_position.zero_grad()

        # Apply the latest attack
        adv_inputs = attack(inputs)
        outputs = model(
            inputs_embeds=adv_inputs["inputs_embeds"],
            labels=adv_inputs["labels"],
            attention_mask=adv_inputs["attention_mask"],
        )
        loss = outputs["loss"]
        pbar.set_postfix({"loss": loss.item()})

        # Backprop
        loss.backward()

        # Optimizer
        with torch.no_grad():
            # Update the closure inputs
            closure_inputs.update(inputs)
            # Alternating optimization
            if (step + 1) % position_cadence == 0:
                optimizer_position.step(closure)
            else:
                optimizer_tokens.step(closure)

        if step == 0 or (step + 1) % 10 == 0:
            # Deterministically generate a response using the adversarial prompt
            prompt_end = adv_inputs["response_mask"].nonzero()[0, -1]
            result = model.generate(
                inputs=adv_inputs["input_ids"][:, :prompt_end],
                attention_mask=adv_inputs["attention_mask"][:, :prompt_end],
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=wrapped_tokenizer.pad_token_id,
            )
            result = wrapped_tokenizer.decode(result[0])
            print(f"{result = }")


if __name__ == "__main__":
    fire.Fire(attack)
