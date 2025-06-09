#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import torch
from transformers import (
    TextClassificationPipeline,
    AutoModelForSequenceClassification,
    BatchEncoding,
)
from transformers.pipelines import PIPELINE_REGISTRY

from .. import TaggedTokenizer, AttackPrompt, AdversarialAttack


class AdversarialTextClassificationPipeline(TextClassificationPipeline):
    def __init__(
        self,
        *args,
        attack: AttackPrompt,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert self.framework == "pt"

        self.model.requires_grad_(False)
        self.apply_attack_template = attack

        self.tokenizer = TaggedTokenizer(
            self.tokenizer, self.apply_attack_template.tags
        )

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

    def _sanitize_parameters(self, *args, label=None, **kwargs):
        preprocess_kwargs, forward_kwargs, postprocess_kwargs = (
            super()._sanitize_parameters(*args, **kwargs)
        )

        if label is not None:
            preprocess_kwargs["label"] = label

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs, **preprocess_kwargs):
        label = preprocess_kwargs.pop("label", None)
        if label is None:
            raise ValueError("You must supply a label")

        adv_inputs = self.apply_attack_template(inputs)

        model_inputs = super().preprocess(adv_inputs, **preprocess_kwargs)
        assert isinstance(model_inputs, BatchEncoding)
        model_inputs["labels"] = torch.tensor(
            [self.model.config.label2id[label]], dtype=torch.long
        )
        return model_inputs

    def get_inference_context(self):  # type: ignore
        return torch.enable_grad

    def _ensure_tensor_on_device(self, inputs, device):
        # Always keep stuff on self.device
        device = self.device
        return super()._ensure_tensor_on_device(inputs, device)

    def _forward(
        self, model_inputs, inf_loss_when_nonreencoding: bool = True, **forward_kwargs
    ):
        adv_model_inputs = self.attack(model_inputs)

        model_outputs = self.model(
            inputs_embeds=adv_model_inputs["inputs_embeds"],
            attention_mask=adv_model_inputs["attention_mask"],
            labels=adv_model_inputs["labels"],
            **forward_kwargs,
        )

        if inf_loss_when_nonreencoding and not self.tokenizer.reencodes(
            adv_model_inputs["input_ids"], adv_model_inputs["attention_mask"]
        ):
            model_outputs["loss"].fill_(torch.inf)

        return adv_model_inputs | model_outputs

    def postprocess(self, model_outputs, *args, **postprocess_kwargs):
        input_ids = model_outputs["input_ids"]
        attention_mask = model_outputs["attention_mask"]

        model_outputs["text"] = self.tokenizer.decode(  #  type: ignore
            input_ids[attention_mask == 1], skip_special_tokens=True
        )

        return model_outputs


PIPELINE_REGISTRY.register_pipeline(
    "adv-text-classification",
    pipeline_class=AdversarialTextClassificationPipeline,
    pt_model=AutoModelForSequenceClassification,
)
