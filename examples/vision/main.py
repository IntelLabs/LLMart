#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
# type: ignore

import fire  # type: ignore[reportMissingImports]
import torch
import requests
from tqdm import tqdm
from PIL import Image

from transformers import MllamaForConditionalGeneration, AutoProcessor

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
    # Define user input and image
    user_input = "If I had to write a haiku for this one"

    user_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    user_image = Image.open(requests.get(user_url, stream=True).raw)

    induced_response = "Self-attention is not supported in PyTorch.<|eot_id|>"

    # Get HF model and processor (contains tokenizer)
    model_id = "meta-llama/Llama-3.2-11B-Vision"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="bfloat16",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # Get processor.tokenizer and wrap it
    tokenizer = processor.tokenizer
    tokenizer.clean_up_tokenization_spaces = False
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Create transforms that inject markers into prompts
    add_attack = AttackPrompt(suffix=suffix)
    force_response = MaskCompletion(replace_with=induced_response)

    # Make tokenizer aware of attack marker tokens
    wrapped_tokenizer = TaggedTokenizer(
        tokenizer, tags=add_attack.tags + force_response.tags
    )
    # Update processor with wrapped tokenizer
    processor.tokenizer = wrapped_tokenizer

    # Create and tokenize conversation
    conversation = [
        dict(role="user", content=add_attack(user_input)),
        dict(role="assistant", content=force_response("")),
    ]

    # Prepare Inputs
    processor.chat_template = conversation
    text_inputs = processor(text=user_input, return_tensors="pt")
    image_inputs = processor(images=user_image, return_tensors="pt")
    inputs = {**text_inputs, **image_inputs}

    # Convert inputs to tensors
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    pixel_values = inputs["pixel_values"].to(model.device)

    ''' TODO - Able to convert processor from AutoProcessor class into wrapped_tokenizer, and getting the `inputs` dictionary containing both text_inputs and image_inputs, but somehow the attack does not proceed further due to the following error:
    torch.cat(): expected a non-empty list of Tensors
    File "/home/adarshan/LLMart/src/llmart/model.py", line 98, in _create_parameters
        param = torch.nn.Parameter(torch.cat(list(params.values()), dim=0))
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/adarshan/LLMart/src/llmart/model.py", line 52, in __init__
        self.param, self.slices = self._create_parameters(attacks)
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/adarshan/LLMart/examples/vision/main.py", line 84, in attack
        token_attack = AdversarialAttack(inputs, model.get_input_embeddings()).to(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/adarshan/LLMart/examples/vision/main.py", line 162, in <module>
        fire.Fire(attack)
    RuntimeError: torch.cat(): expected a non-empty list of Tensors

    NOTE - Due to less time and lack of understanding of the code, I am unable to debug the error, so submitting the partial code as it is.
    '''
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