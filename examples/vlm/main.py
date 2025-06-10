#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import copy
import random
import requests
import tempfile

from tqdm import tqdm
from tqdm.auto import trange
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.v2.functional import pil_to_tensor, to_pil_image
from transformers import pipeline, AutoProcessor

# Import image attack components from mart.
from mart.attack.initializer import Uniform  # pyright: ignore [reportMissingImports]
from mart.attack.projector import Lp, Range, Compose  # pyright: ignore [reportMissingImports]
from mart.attack.perturber import UniversalPerturber  # pyright: ignore [reportMissingImports]
from mart.attack.composer.patch import PertImageBase  # pyright: ignore [reportMissingImports]

# Register the adversarial pipeline when importing anything from llmart.
from data import ComputerUseAgenticAIDataset


# Adversarial composition
class AdversarialComposer(torch.nn.Module):
    def __init__(
        self,
        *,
        init_image_fpath,
        x: int = 236,
        y: int = 420,
        w: int = 1180,
        h: int = 614,
    ):
        super().__init__()
        # Define adversarial perturbation in the [0,255] space, making it easier to understand the step size.
        # Initialize as [-6,6] uniform integers.
        uniform_round_initializer = Uniform(min=-6, max=6, round=True)
        # zero_initializer = mart.attack.initializer.Constant(0)

        # Project the perturbation to the L2 bounded integers.
        l2_projector = Lp(p=2, eps=4423)
        range_round_projector = Range(quantize=True, min=-255, max=255)
        composed_projector = Compose([l2_projector, range_round_projector])

        # A pertuber object that contains the rounded perturbation.
        self.perturber = UniversalPerturber(
            shape=(3, h, w),
            initializer=uniform_round_initializer,
            # initializer=zero_initializer,
            projector=composed_projector,
            # Gradient sign, making learning_rate == step_size for SGD.
            grad_modifier=lambda x: x.sign(),
        )

        # Add perturbation to the image in the [0, 255] space.
        self.image_additive = PertImageBase(fpath=init_image_fpath)
        self.x, self.y, self.w, self.h = x, y, w, h

    def forward(self, inputs_tensor: torch.Tensor, return_dict: bool = False):
        outputs = {}
        # Perturbation in the [0,255] space.
        # FIXME: Project requires input, though we don't actually need it for the projection here.
        perturbation = self.perturber(input=inputs_tensor, target=None)
        outputs["perturbation"] = perturbation

        patch_perturbed_255 = self.image_additive(perturbation)
        outputs["patch_perturbed_255"] = patch_perturbed_255

        input_adv_255 = inputs_tensor.clone().to(patch_perturbed_255.dtype)
        # NCHW
        input_adv_255[..., self.y : self.y + self.h, self.x : self.x + self.w] = (
            patch_perturbed_255
        )

        outputs["input_adv_255"] = input_adv_255

        if return_dict:
            # Output a dictionary to visualize intermediate images.
            return outputs
        else:
            return input_adv_255


def replace_image(prompt, adv_image_composer):
    """Replace the image in a prompt and return a new prompt."""
    prompt = copy.deepcopy(prompt)
    adv_outputs = []
    for msg in prompt:
        if isinstance(msg["content"], list) and msg["content"][0]["type"] == "image":
            for content in msg["content"]:
                pil_image = content["image"].convert("RGB")
                image_tensor = pil_to_tensor(pil_image)
                adv_output = adv_image_composer(image_tensor)
                # PyTorch assumes [0,1] range for float image tensor.
                pil_image_adv = to_pil_image(adv_output.to(torch.uint8))
                content["image"] = pil_image_adv
                adv_outputs.append(adv_output)
    return prompt, adv_outputs


def main(
    model: str = "ByteDance-Seed/UI-TARS-2B-SFT",
    processor: str | None = None,
    revision: str = "f366a1db3e7f29635f5b236d6a71dea367a0a700",
    init_image_url: str = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg/960px-Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg",
    max_pixels: int = 1024 * 28 * 28,
    max_epochs: int = 1000,
    logdir: str = "logs/UI_TARS_2B_adv_screenshot",
    seed: int = 2025,
):
    # Program entry.
    # Seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # 1. Get a pipeline of the target model, using parameters from the UI_TARS app.
    processor = processor or model

    pipe = pipeline(  # pyright: ignore [reportArgumentType]
        "image-text-to-text",
        model=model,
        processor=AutoProcessor.from_pretrained(
            processor,
            max_pixels=max_pixels,
            size=None,  # Newer transformers redefine the keys in size, so we make it empty and fallback to min_pixels and max_pixels.
            use_fast=True,  # Use fast (and differentiable) image processing.
        ),
        revision=revision,
        torch_dtype=torch.bfloat16,
        generate_kwargs=dict(
            do_sample=False,
            max_new_tokens=1000,
            top_p=None,
            temperature=None,
        ),
    )

    assert pipe.model.generation_config is not None
    # Why do we add pad_token_id here?
    pipe.model.generation_config.pad_token_id = (
        pipe.model.generation_config.eos_token_id
    )

    # Enable pipe.save_pretrained
    # KeyError: 'torch_dtype' in transformers v4.50.3
    # del pipe.tokenizer.init_kwargs["torch_dtype"]  # pyright: ignore [reportOptionalMemberAccess]

    pipe.model.eval()
    pipe.model.requires_grad_(False)

    # 2. Load the dataset
    dataset = ComputerUseAgenticAIDataset()
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_data = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda batch: batch[0],
        generator=generator,
    )

    response = pipe(text=dataset[0]["prompt"])  # pyright: ignore [reportCallIssue]
    print(f"Original response: {response}")

    # 3. Prepare the adversary pipeline for the target model.
    # Download and load the starting image.
    with tempfile.NamedTemporaryFile() as fp:
        img_data = requests.get(
            init_image_url,
            headers={"User-Agent": "LLMart (https://github.com/IntelLabs/llmart)"},
        ).content
        fp.write(img_data)
        # FIXME: Let MART accept PIL image too.
        image_attack = AdversarialComposer(
            init_image_fpath=fp.name, x=883, y=239, w=575, h=709
        )

    adv_pipe = pipeline(  # pyright: ignore [reportArgumentType]
        task="adv-image-text-to-loss",
        model=pipe.model,
        # do not reuse the processor, because we will add the TaggedTokenizer in the processor.
        processor=AutoProcessor.from_pretrained(
            processor,
            max_pixels=max_pixels,
            size=None,  # newer transformers redefine the keys in size, so we make it empty and fallback to min_pixels and max_pixels.
            use_fast=True,  # use fast (and differentiable) image processing.
        ),
        image_attack=image_attack,
    )

    # 4. Optimize the perturbation
    optimizer = torch.optim.Adam(adv_pipe.image_attack.parameters(), lr=1)  # pyright: ignore [reportAttributeAccessIssue]

    # TODO: Get rid of tensorboard, later.
    logger = SummaryWriter(logdir, flush_secs=1)

    generate_every = 20

    epoch_loss = -1
    with trange(max_epochs, desc="Epoch") as t:
        for epoch in t:
            t.set_description(f"Epoch {epoch}")
            t.set_postfix(loss=epoch_loss)

            if epoch % generate_every == 0:
                with torch.inference_mode():
                    # TODO: You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset. 31 secs now.
                    for i, inputs in enumerate(
                        tqdm(dataset, leave=True, desc="generate")  # pyright: ignore [reportCallIssue, reportArgumentType]
                    ):
                        # Retrieve the adversarial image, to replace the image in the dataset.
                        prompt, adv_screenshots = replace_image(
                            inputs["prompt"],
                            adv_pipe.image_attack,  # pyright: ignore [reportAttributeAccessIssue]
                        )
                        outputs = pipe(text=prompt)  # pyright: ignore[reportCallIssue]
                        response = outputs[0]["generated_text"][len(prompt) :]
                        # TODO: Compare with the expected response at the end of conversation, and calcualte the success rate.
                        logger.add_text(
                            f"response_{i}", str(response), global_step=epoch
                        )
                        logger.add_image(
                            "adv_screenshot",
                            adv_screenshots[0] / 255,  # pyright: ignore[reportOptionalOperand]
                            global_step=epoch,
                        )

            epoch_losses = []
            for i, inputs in enumerate(
                pbar2 := tqdm(train_data, total=len(dataset), leave=False, desc="batch")
            ):
                optimizer.zero_grad()
                outputs = adv_pipe(  # pyright: ignore[reportCallIssue]
                    text=inputs["conversation"]
                )
                loss = outputs["outputs"].loss
                loss.backward()
                optimizer.step()

                loss_float = float(loss.cpu().detach())
                epoch_losses.append(loss_float)
                logger.add_scalar(
                    "loss", loss_float, global_step=i + epoch * len(dataset)
                )
                pbar2.set_postfix(
                    {
                        "loss": f"{epoch_losses[-1]:0.4f}",
                    }
                )
            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            logger.add_scalar("epoch_loss", epoch_loss, global_step=epoch)


if __name__ == "__main__":
    import fire  # pyright: ignore [reportMissingImports]

    fire.Fire(main)
