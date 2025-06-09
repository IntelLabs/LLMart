#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#


import torch
from torchvision.transforms.v2.functional import pil_to_tensor

from transformers import ImageTextToTextPipeline
from transformers import AutoModelForCausalLM
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.image_utils import make_flat_list_of_images

from llmart import TaggedTokenizer


class AdvImageProcessor(torch.nn.Module):
    """ "Compose images with perturbation before running the image_processor."""

    def __init__(self, image_processor, image_attack, device="cpu"):
        super().__init__()

        self.device = device
        self.image_attack = image_attack
        self.image_processor = image_processor

        # Move the perturbation parameter to the right device.
        self.image_attack.to(self.device)

    def __getattr__(self, name):
        if hasattr(self.image_processor, name):
            # Expose attributes in the original image processor.
            return getattr(self.image_processor, name)
        else:
            return super().__getattr__(name)

    def forward(self, *args, images, **kwargs):
        # We may have nested images.
        images = make_flat_list_of_images(images)
        # Convert PIL images to NCWH tensor in uint8.
        images_tensor = torch.stack([pil_to_tensor(image) for image in images])  # pyright: ignore [reportOptionalIterable, reportGeneralTypeIssues]
        # So we will run image processing on the device.
        images_tensor = images_tensor.to(self.device)
        # Adversarial image composition.
        adv_images_tensor = self.image_attack(images_tensor)
        outputs = self.image_processor(*args, images=adv_images_tensor, **kwargs)
        return outputs


class AdversarialImageTextToLossPipeline(ImageTextToTextPipeline):
    """A customized pipeline for adversary to get the model loss instead of generating responses."""

    def _init_image_attack(self, image_attack):
        """Preparation for image attack."""

        if self.framework != "pt":
            raise RuntimeError(f"{self.__class__.__name__} only runs on PyTorch.")

        self.image_attack = image_attack

        # The TaggedTokenizer allows us to locate the response for generating labels in computing the loss.
        tokenizer = TaggedTokenizer(
            self.processor.tokenizer,  # pyright: ignore [reportAttributeAccessIssue, reportOptionalMemberAccess]
            tags=["<|begin_response|>", "<|end_response|>"],
        )
        setattr(self.processor, "tokenizer", tokenizer)

        # Wrap self.processor.image_processor.
        image_processor = AdvImageProcessor(
            self.processor.image_processor,  # pyright: ignore [reportAttributeAccessIssue, reportOptionalMemberAccess]
            self.image_attack,
            device=self.device,  # pyright: ignore [reportArgumentType]
        )
        setattr(self.processor, "image_processor", image_processor)

        self.model.requires_grad_(False)

    def _sanitize_parameters(  # pyright: ignore [reportIncompatibleMethodOverride]
        self, *, image_attack=None, **kwargs
    ):
        # We expect users to pass image_attack and pytorch_image_transform in __init__(), but not in __call__().
        if image_attack is not None:
            self._init_image_attack(image_attack)

        return super()._sanitize_parameters(**kwargs)

    def get_inference_context(self):  # pyright: ignore [reportIncompatibleMethodOverride]
        # Override the inference context, because the pipeline outputs a loss for the backward pass.
        return torch.enable_grad

    def _ensure_tensor_on_device(self, inputs, device):
        # Always keep stuff on self.device. Avoid moving the forward results to CPU.
        device = self.device
        return super()._ensure_tensor_on_device(inputs, device)

    def _forward(self, model_inputs, generate_kwargs=None):
        # Get labels to compute loss.
        response_mask = model_inputs.pop("response_mask")
        labels = torch.where(response_mask, model_inputs["input_ids"], -100)
        labels = labels.to(self.device, non_blocking=True)

        # Pop some keys to return because model.forward() will complain about the unknown keys.
        ret = {}
        allowed_keys = ["input_ids", "pixel_values", "image_grid_thw", "attention_mask"]
        for key in list(model_inputs.keys()):
            if key not in allowed_keys:
                ret[key] = model_inputs.pop(key)

        outputs = self.model(**model_inputs, labels=labels)
        ret["outputs"] = outputs
        return ret

    def postprocess(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        model_outputs,
        **kwargs,
    ):
        # No postprocessing for computing the loss.
        return model_outputs


PIPELINE_REGISTRY.register_pipeline(
    "adv-image-text-to-loss",
    pipeline_class=AdversarialImageTextToLossPipeline,
    pt_model=AutoModelForCausalLM,
)
