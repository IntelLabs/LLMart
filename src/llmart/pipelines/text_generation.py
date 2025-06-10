#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import torch
from warnings import warn
from transformers import TextGenerationPipeline, AutoModelForCausalLM, BatchEncoding
from transformers.pipelines.text_generation import Chat
from transformers.pipelines import PIPELINE_REGISTRY

from .. import TaggedTokenizer, AttackPrompt, AdversarialAttack


class AdversarialTextGenerationPipeline(TextGenerationPipeline):
    def __init__(self, *args, attack: AttackPrompt, **kwargs):
        super().__init__(*args, **kwargs)
        if self.framework != "pt":
            raise ValueError(
                f"{self.__class__.__name__} only supports the 'pt' framework!"
            )

        self.model.requires_grad_(False)
        self.apply_attack_template = attack

        self.tokenizer = TaggedTokenizer(
            self.tokenizer, self.apply_attack_template.tags
        )

        if self.tokenizer.pad_token is None:
            warn("Setting pad_token to eos_token!")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.attack = AdversarialAttack(
            self.tokenizer(
                self.apply_attack_template.elements,
                add_special_tokens=False,
                padding=True,
                return_tensors=self.framework,
            ),
            self.model.get_input_embeddings(),
        )

    def __call__(self, *args, **kwargs):
        # supress pipeline call count warning
        self.call_count = 0
        return super().__call__(*args, **kwargs)

    def _sanitize_parameters(self, *args, completion=None, **kwargs):
        preprocess_params, forward_params, postprocess_params = (
            super()._sanitize_parameters(*args, **kwargs)
        )

        if completion is not None:
            preprocess_params["completion"] = completion

        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, prompt_text, *args, completion=None, **kwargs):
        if isinstance(prompt_text, Chat):
            kwargs.pop("continue_final_message", None)

            messages = prompt_text.messages
            if messages[-1]["role"] != "assistant":
                raise ValueError("Final message but be an assistant message")
            messages = self.apply_attack_template(messages)

            inputs: BatchEncoding = super().preprocess(  # type: ignore
                Chat(messages),
                *args,
                **kwargs,
                continue_final_message=True,
            )

            prompts: BatchEncoding = super().preprocess(  # type: ignore
                Chat(messages[:-1]),
                *args,
                **kwargs,
                continue_final_message=False,
            )
        else:
            if completion is None:
                raise ValueError("You must supply a completion.")

            # Apply attack template to prompt
            adv_prompt_text = self.apply_attack_template(prompt_text)

            inputs: BatchEncoding = super().preprocess(  # type: ignore
                adv_prompt_text + completion, *args, **kwargs
            )

            prompts: BatchEncoding = super().preprocess(  # type: ignore
                adv_prompt_text, *args, **kwargs
            )

        # Compute assistant and prompt masks
        inputs_ids = inputs["input_ids"]
        assert isinstance(inputs_ids, torch.Tensor)
        prompts_ids = prompts["input_ids"]
        assert isinstance(prompts_ids, torch.Tensor)

        assistant_mask = torch.zeros_like(inputs_ids, dtype=torch.bool)
        prompt_mask = torch.zeros_like(inputs_ids, dtype=torch.bool)
        for i, prompt_ids in enumerate(prompts_ids):
            assistant_mask[i, len(prompt_ids) :] = True
            prompt_mask[i, : len(prompt_ids)] = True

        prompt_text = inputs.pop("prompt_text")

        # Move inputs to proper device and attack them
        model_inputs = self.ensure_tensor_on_device(**inputs)
        model_inputs = self.attack(model_inputs)

        model_inputs["reencodes"] = self.tokenizer.reencodes(
            model_inputs["input_ids"], model_inputs["attention_mask"]
        )
        model_inputs["prompt_text"] = prompt_text
        model_inputs["prompt_mask"] = prompt_mask
        model_inputs["assistant_mask"] = assistant_mask

        return model_inputs

    def get_inference_context(self):  # type: ignore
        return torch.enable_grad

    def _ensure_tensor_on_device(self, inputs, device):
        # Always keep stuff on self.device
        device = self.device
        return super()._ensure_tensor_on_device(inputs, device)

    def _forward(
        self,
        model_inputs,
        inf_loss_when_nonreencoding: bool = True,
        **forward_kwargs,
    ):
        prompt_text = model_inputs.pop("prompt_text")

        prompt_mask = model_inputs["prompt_mask"]
        inputs_embeds = model_inputs.get("inputs_embeds", None)
        attention_mask = model_inputs.get("attention_mask", None)

        # Fabricate labels from assistant mask
        input_ids = model_inputs["input_ids"]
        assistant_mask = model_inputs["assistant_mask"]
        labels = input_ids.clone()
        labels[~assistant_mask] = -100

        # FIXME: Add option to use generate with return_logits=True
        model_kwargs = dict(
            attention_mask=attention_mask, labels=labels, **forward_kwargs
        )
        if inputs_embeds is not None:
            model_kwargs["inputs_embeds"] = inputs_embeds
        else:
            model_kwargs["input_ids"] = input_ids

        outputs = self.model(**model_kwargs)
        logits = outputs["logits"]
        loss = outputs["loss"]

        # Compute per-example loss
        losses = torch.vmap(torch.nn.functional.cross_entropy)(
            logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
        )
        if inf_loss_when_nonreencoding:
            reencodes = model_inputs["reencodes"]
            losses = torch.where(reencodes, losses, torch.inf)
            loss = torch.where(reencodes.all(), loss, torch.inf)

        # Combine input ids with predicted logits
        # NOTE: We assume there is at least one token in the prompt because we
        #       roll the logits to account for causality.
        generated_sequence = torch.where(
            assistant_mask, logits.argmax(-1).roll(1, dims=-1), input_ids
        )[None]

        return {
            "generated_sequence": generated_sequence,
            "input_ids": input_ids,
            "prompt_text": prompt_text,
            "prompt_mask": prompt_mask,
            "logits": logits,
            "labels": labels,
            "loss": loss,
            "losses": losses,
        }

    def postprocess(self, model_outputs, *args, **kwargs):
        # Mask out non-prompt tokens
        input_ids = model_outputs.pop("input_ids")
        prompt_mask = model_outputs.pop("prompt_mask")
        prompt_ids = input_ids[prompt_mask][None]
        model_outputs["input_ids"] = prompt_ids

        # Remove last assistant chat from input_ids
        prompt_text = model_outputs.pop("prompt_text")
        if isinstance(prompt_text, Chat):
            # Re-encode chat contents to include adversarial token replacements and
            # remove last assistant chat
            for message in prompt_text.messages[:-1]:
                inputs = self.tokenizer(
                    message["content"],
                    add_special_tokens=False,
                    return_tensors=self.framework,
                )
                inputs = self.ensure_tensor_on_device(**inputs)
                adv_inputs = self.attack(inputs)
                message["content"] = self.tokenizer.decode(adv_inputs["input_ids"][0])
            model_outputs["prompt_text"] = Chat(prompt_text.messages[:-1])

        else:
            model_outputs["prompt_text"] = self.tokenizer.decode(
                model_outputs["input_ids"][0], skip_special_tokens=True
            )

        # Super assumes generated_sequence is on cpu by calling numpy on it
        model_outputs["generated_sequence"] = model_outputs["generated_sequence"].cpu()
        records = super().postprocess(model_outputs, *args, **kwargs)
        assert len(records) == 1

        # Add loss and prompt to output
        records[0]["prompt_text"] = model_outputs["prompt_text"]
        records[0]["loss"] = model_outputs["loss"].squeeze()
        records[0]["input_ids"] = model_outputs["input_ids"].squeeze()
        records[0]["logits"] = model_outputs["logits"].squeeze()
        records[0]["labels"] = model_outputs["labels"].squeeze()
        return records


PIPELINE_REGISTRY.register_pipeline(
    "adv-text-generation",
    pipeline_class=AdversarialTextGenerationPipeline,
    pt_model=AutoModelForCausalLM,
)
