#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import torch
from tqdm import trange
from torch.optim import Adam  # type: ignore[reportPrivateImportUsage]
from collections.abc import MutableMapping
from transformers import pipeline, PreTrainedTokenizerBase, Pipeline
from transformers.pipelines.text_generation import ReturnType
from llmart import (
    AttackPrompt,
    AdversarialTextGenerationPipeline,
    GreedyCoordinateGradient,
)


def main(
    prompt: str,
    completion: str,
    max_steps: int,
    max_optim_tokens: int,
    lr: float,
    use_hard_tokens: bool,
):
    generator = pipeline(
        task="text-generation",
        model="microsoft/Llama2-7b-WhoIsHarryPotter",
        revision="e4347ee6643edd19b107a120a80388e126d407c0",
        # model = "meta-llama/Llama-2-7b-chat-hf",
        # revision= "f5db02db724555f92da89c216ac04704f23d4590",
        device_map="auto",
        do_sample=False,
        top_p=None,
        temperature=None,
        max_new_tokens=50,
        model_kwargs=dict(local_files_only=True),
        return_type=ReturnType.NEW_TEXT,
    )
    assert isinstance(generator.tokenizer, PreTrainedTokenizerBase)
    generator.tokenizer.pad_token = generator.tokenizer.eos_token

    prefix_len = 1
    while prefix_len <= max_optim_tokens:
        print(
            f"\n Searching for '{prompt}' => '{completion}' | with: prefix_len({prefix_len})"
        )
        found, (adv_prefix, gen_completion) = attack(
            prompt,
            generator=generator,
            completion=completion,
            prefix_len=prefix_len,
            max_steps=max_steps,
            lr=lr,
            use_hard_tokens=use_hard_tokens,
        )
        print(f"Final prompt: {repr(adv_prefix)} => {repr(gen_completion)}")
        if found:
            print("Found effective prompt")
            break
        else:
            print("Failed to find effective prompt")
            prefix_len *= 2


def attack(
    prompt: str,
    *,
    generator: Pipeline,
    completion: str,
    prefix_len: int,
    max_steps: int,
    lr: float,
    use_hard_tokens: bool,
    seed: int = 2024,
) -> tuple[bool, tuple[str, str]]:
    torch.manual_seed(seed)
    adv_generator = pipeline(
        task="adv-text-generation",
        model=generator.model,
        tokenizer=generator.tokenizer,
        attack=AttackPrompt(prefix=prefix_len, default_token=" @"),
        inf_loss_when_nonreencoding=use_hard_tokens,
        return_type=ReturnType.NEW_TEXT,
    )
    assert isinstance(adv_generator, AdversarialTextGenerationPipeline)

    optim = (
        GreedyCoordinateGradient(
            adv_generator.attack.parameters(),
            ignored_values=adv_generator.tokenizer.bad_token_ids,
            negative_only=False,
            n_tokens=1,
            coord_randk=0,
            coord_topk=256,
            global_topk=0,
        )
        if use_hard_tokens
        else Adam(adv_generator.attack.parameters(), lr=lr)
    )

    adv_prompt = prompt
    adv_completion = ""
    found = False

    for _ in (pbar := trange(max_steps)):
        # Check if we found an attack that works
        with torch.inference_mode():
            found = adv_completion.startswith(completion)
            if found:
                break

        def closure(return_outputs=False):
            outputs: MutableMapping = adv_generator(prompt, completion=completion)[0]  # type: ignore
            return outputs if return_outputs else outputs["loss"]

        # Check if we found an attack that works
        adv_outputs: MutableMapping = closure(return_outputs=True)
        adv_prompt = adv_outputs["prompt_text"]
        adv_completion = adv_outputs["generated_text"]
        loss = adv_outputs["loss"]

        loss.backward()
        with torch.inference_mode():
            if use_hard_tokens:
                optim.step(closure)
            else:
                optim.step()  # type: ignore

        pbar.set_postfix(loss=f"{loss:0.4f}")

    # Manually pass inputs_embeds to original generator to double check
    if use_hard_tokens:
        output: MutableMapping = generator(adv_prompt)[0]  # type: ignore
        decoded = output["generated_text"]
    else:
        model_inputs = adv_generator.preprocess(prompt, completion="")  # type: ignore
        model_inputs = adv_generator.ensure_tensor_on_device(**model_inputs)
        assert isinstance(adv_generator, AdversarialTextGenerationPipeline)
        adv_model_inputs = adv_generator.attack(model_inputs)
        output_ids = generator.model.generate(
            inputs_embeds=adv_model_inputs["inputs_embeds"], max_length=100
        )[0]
        decoded = generator.tokenizer.decode(output_ids)  # type: ignore

    return found, (adv_prompt, decoded)  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        type=str,
        help="Prompt provided to the model.",
    )
    parser.add_argument(
        "completion",
        type=str,
        help="Target completion.",
    )
    parser.add_argument(
        "--max_steps",
        dest="max_steps",
        type=int,
        default=5000,
        help="Optimise prompt for no more than N steps.",
    )
    parser.add_argument(
        "--max_optim_tokens",
        dest="max_optim_tokens",
        type=int,
        default=32,
        help="If optimisation fails, increase prompt size until N tokens.",
    )
    parser.add_argument(
        "--lr",
        dest="lr",
        type=float,
        default=0.0005,
        help="Learning rate for adversarial optimisation.",
    )
    parser.add_argument(
        "--use_hard_tokens",
        dest="use_hard_tokens",
        action="store_true",
        help="Find hard tokens instead of soft tokens in the emebdding space.",
    )

    args = parser.parse_args()
    if args.use_hard_tokens:
        print("WARN! Optimising hard tokens; lr will have no effect")

    main(
        args.prompt,
        args.completion,
        args.max_steps,
        args.max_optim_tokens,
        args.lr,
        args.use_hard_tokens,
    )
