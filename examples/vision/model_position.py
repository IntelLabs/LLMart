#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn.functional as F
from typing import Any
from collections.abc import Mapping


class AdversarialBlockShift(torch.nn.Module):
    """Implements adversarial token block position shifting in user-controllable inputs.

    Uses convolution-based shifting of the token embeddings to preserve gradients.

    Args:
        attacks: Dictionary containing attack masks and response masks
        embedding: Token embedding dictionary
        extra_tokens: List of [preamble, postamble] token counts that are not modifiable by the user.

    """

    def __init__(
        self,
        attacks: Mapping[str, Any],
        embedding: torch.nn.Module,
        # FIXME: This is a Llama 3 workaround for currently not having a "front-end" user mask
        extra_tokens: list[int] = [5, 5],  # [preamble, postamble]
    ):
        super().__init__()

        if not isinstance(embedding, torch.nn.Embedding):
            raise ValueError("embedding must be an nn.Embedding")
        self._embedding = [embedding]
        self.extra_tokens = extra_tokens

        # Get the allowable block mask and maximum block shifts in both directions
        (
            self.mask_name,
            self.frontend_user_mask,
            self.max_left_shift,
            self.max_right_shift,
        ) = self._get_move_mask_and_shifts(attacks, extra_tokens)

        # Instantiate the parameters as an asymmetric kernel, and padding for it
        self.param = self._create_parameters()

    def _get_move_mask_and_shifts(
        self, attack_init: Mapping[str, Any], extra_tokens: list[int]
    ) -> tuple[str, torch.Tensor, int, int]:
        mask_name = self._find_attack_key(attack_init)
        # Get the position of the original suffix in the user-controllable (front-end) tokens
        # NOTE: The mask is intended to only cover the latest (in a multi-turn conversation)
        # user-controllable message
        response_begin = torch.where(attack_init["response_mask"])[-1][0]
        frontend_user_mask = attack_init["response_mask"].logical_not()
        frontend_user_mask[..., : extra_tokens[0]] = False
        frontend_user_mask[..., response_begin - extra_tokens[1] :] = False
        block_mask = attack_init[mask_name]
        move_mask = torch.cat(
            [
                b_mask[fe_mask]
                for b_mask, fe_mask in zip(block_mask, frontend_user_mask)
            ],
            dim=0,
        )

        # Compute the maximum realizable shifts
        max_left_shift = int(torch.where(move_mask)[-1][0].item())
        max_right_shift = int(
            len(move_mask) - (torch.where(move_mask)[-1][-1] + 1).item()
        )

        return mask_name, frontend_user_mask, max_left_shift, max_right_shift

    def _find_attack_key(self, attack_init: Mapping[str, Any]):
        # Store the attack mask name
        found, mask_name = False, None
        for key in attack_init:
            if key in ["prefix_mask", "suffix_mask"] and found:
                raise ValueError(
                    "Optimizing adversarial block position doesn't currently support simultaneous prefix and suffix!"
                )

            if key in ["prefix_mask", "suffix_mask"]:
                found = True
                mask_name = key

        if not mask_name:
            raise ValueError(
                "Optimizing adversarial block position requires attack=suffix or attack=prefix!"
            )

        return mask_name

    def _create_parameters(self):
        # One-hot convolutional kernel that shifts the attack mask
        # Interpretation **after** padding and reflection:
        # (0, 0, ..., 0, 1, 0, ..., 0, 0) (1 in center) -> adversarial block stays in place
        # (0, 0, ..., 1, 0, 0, ..., 0, 0) -> adversarial block shifts one position to the LEFT
        # (0, 0, ..., 0, 0, 1, ..., 0, 0) -> adversarial block shifts one position to the RIGHT
        param = torch.nn.Parameter(
            torch.zeros(
                1,
                self.max_left_shift + self.max_right_shift + 1,
                dtype=torch.float32,
            )
        )

        # Initialize to centered Dirac impulse (after padding)
        param.data[0, self.max_left_shift] = 1.0

        # Compute amount of padding for the differentiable kernel
        self.left_pad = int(max(0, self.max_right_shift - self.max_left_shift))
        self.right_pad = int(max(0, self.max_left_shift - self.max_right_shift))
        self.padded_kernel_size = 2 * max(self.max_left_shift, self.max_right_shift) + 1

        return param

    @property
    def embedding(self):
        return self._embedding[0]

    def _embed(self, input_ids):
        return self.embedding(input_ids.to(device=self.embedding.weight.device))

    def forward(self, inputs):
        """Shifts adversarial blocks within the sequence using current parameters.

        Args:
            inputs: Dictionary containing 'input_ids', user mask, and optionally 'inputs_embeds'

        Returns:
            Dictionary with modified 'input_ids' and 'inputs_embeds'
        """
        # Re-use token embeddings were already computed
        batch_inputs_embeds = inputs.pop(
            "inputs_embeds", self._embed(inputs["input_ids"])
        )

        output_embeds, output_ids = [], []
        for attack_mask, inputs_embeds, input_ids, frontend_mask in zip(
            inputs[self.mask_name],
            batch_inputs_embeds,
            inputs["input_ids"],
            self.frontend_user_mask,
        ):
            # Get the padded kernel and reflect it
            # https://discuss.pytorch.org/t/why-are-pytorch-convolutions-implemented-as-cross-correlations/115010
            padded_kernel = (
                F.pad(self.param, pad=(self.left_pad, self.right_pad, 0, 0))
                .flip(dims=(-1,))
                .to(inputs_embeds.dtype)
            )

            # Extract the embeddings for the frontend user and everything else
            other_embeds = inputs_embeds[frontend_mask.logical_not()]
            frontend_user_embeds = inputs_embeds[frontend_mask]

            # Filter by treating the embedding dimension as "channels"
            # Will cause a warning https://github.com/pytorch/pytorch/issues/47163 with DDP
            new_frontend_user_embeds = (
                F.conv1d(
                    frontend_user_embeds.T[None, ...],
                    # Same filter is applied to all "channels"
                    padded_kernel[None, ...].expand(
                        frontend_user_embeds.shape[-1], -1, -1
                    ),
                    groups=frontend_user_embeds.shape[-1],
                    padding="same",
                )
                .squeeze()
                .T
            )

            # Assemble embeddings back together
            embeddings = torch.cat(
                [
                    other_embeds[: self.extra_tokens[0]],
                    new_frontend_user_embeds,
                    other_embeds[self.extra_tokens[1] :],
                ],
                dim=0,
            )
            output_embeds.append(embeddings)

            # Modify the other inputs (non-differentiably)
            with torch.no_grad():
                current_start = torch.where(
                    attack_mask.isclose(torch.ones_like(attack_mask))
                )[0][0].item()
                mask_shift = (
                    padded_kernel.shape[-1] // 2
                    - torch.where(padded_kernel.squeeze() == 1.0)[0][0].item()
                )
                new_start = current_start + mask_shift

                nonadv_ids = input_ids[~attack_mask]
                adv_ids = input_ids[attack_mask]
                ids = torch.cat(
                    [
                        nonadv_ids[:new_start],
                        adv_ids,
                        nonadv_ids[new_start:],
                    ],
                    dim=0,
                )

                output_ids.append(ids)

        output_embeds = torch.stack(output_embeds)
        output_ids = torch.stack(output_ids)

        inputs["inputs_embeds"] = output_embeds
        inputs["input_ids"] = output_ids

        return inputs
