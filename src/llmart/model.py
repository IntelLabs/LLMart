#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import re
import torch
import torch.nn.functional as F
from typing import Any
from collections.abc import Mapping, MutableMapping
from collections import OrderedDict
from itertools import accumulate

from .tokenizer import MASK_FORMAT, MASK_PATTERN


class AdversarialAttack(torch.nn.Module):
    """Module that contains token attack parameters and applies attack.

    Creates differentiable attacks by embedding input tokens and allowing
    gradient-based optimization of the inputs. Can operate in one-hot (dim=0) or
    embedding (dim=1) space. Calling this modules modifies the input to contain
    the adversarial attack.

    Args:
        attacks: Mapping containing attack specifications including input_ids and masks
        embedding: Token embedding layer to attack
        dim: Dimension to operate in (0=one-hot, 1=embedding space)
        init: Initial attack parameters (tensor, path to saved state, or None)

    Raises:
        ValueError: If embedding is not nn.Embedding or dim is invalid
    """

    def __init__(
        self,
        attacks: Mapping[str, Any],
        embedding: torch.nn.Module,
        dim: int = 0,
        init: str | torch.Tensor | None = None,
    ):
        super().__init__()

        if not isinstance(embedding, torch.nn.Embedding):
            raise ValueError("embedding must be an nn.Embedding")
        if dim not in [0, 1]:
            raise ValueError("dim must be 0 or 1")

        self._embedding = [embedding]
        self.dim = dim
        self.param, self.slices = self._create_parameters(attacks)

        if isinstance(init, torch.Tensor):
            param = init
            if param.shape != self.param.shape:
                # assume set of good token ids to sample from
                input_ids = init[torch.randint(len(init), self.param.shape[0:1])]
                param = self._embed(input_ids)
            self.param.data.copy_(param)
        elif isinstance(init, str):
            attack_state = torch.load(init, weights_only=True, map_location="cpu")
            self.load_state_dict(attack_state, strict=True)

    def _create_parameters(self, attack_init: Mapping[str, Any]):
        if not isinstance(attack_init["input_ids"], torch.Tensor):
            raise ValueError("attack must be a Tensor")

        params = OrderedDict()
        for key in attack_init:
            # TaggedTokenizer adds special "{name}_mask" keys,
            # so we ignore anything else.
            if not (m := re.match(MASK_PATTERN, key)) or key in (
                "attention_mask",
                "response_mask",
            ):
                continue

            # Make sure attack mask is boolean
            attack_mask: torch.BoolTensor = attack_init[key]  # type: ignore
            if attack_mask.dtype != torch.bool:
                continue

            # Make sure attack maps to a single input_map id
            input_map = attack_init["input_map"][attack_mask]
            if len(input_map.unique()) != 1:
                continue

            # Make sure attack is a 1-dimensional tensor
            input_ids = attack_init["input_ids"][attack_mask]
            if len(input_ids.shape) != 1:
                continue

            params[m["name"]] = self._embed(input_ids)

        # Create vector attack parameter by concatenating attacks along sequence dim.
        # Ideally we'd turn self.params into a nn.ParameterDict but optimizers like vectors.
        param = torch.nn.Parameter(torch.cat(list(params.values()), dim=0))

        # Since we flattened, map param names to slices into param
        param_stops = list(accumulate([len(p) for p in params.values()]))
        slices = {
            key: slice(start, stop)
            for key, start, stop in zip(params.keys(), [0, *param_stops], param_stops)
        }
        return param, slices

    def extra_repr(self):
        return torch.nn.ParameterDict(self.params).extra_repr()

    @property
    def params(self) -> dict[int, torch.Tensor]:
        return {key: self.param[s] for key, s in self.slices.items()}

    @property
    def embedding(self):
        return self._embedding[0]

    def _embed(self, input_ids):
        if self.dim == 0:
            return F.one_hot(input_ids, num_classes=self.embedding.num_embeddings).to(
                dtype=self.embedding.weight.dtype,
                device=self.embedding.weight.device,
            )
        return self.embedding(input_ids.to(device=self.embedding.weight.device))

    def forward(self, inputs: MutableMapping[str, Any]) -> Mapping[str, Any]:
        """Applies the adversarial attack to the given input.

        Embeds input token ids into 1-hot or embedding space and scatters the attack
        parameters into those embeddings using the attack masks.

        Args:
            inputs: Mapping containing input_ids and masks for each attack.

        Returns:
            Mapping containing adversarial input_ids and inputs_embeds. You should pass
            inputs_embeds to the model.
        """

        if not torch.is_grad_enabled() and self.dim == 0:
            return self._fast_forward(inputs)
        else:
            return self._forward(inputs)

    def _forward(self, inputs: MutableMapping[str, Any]) -> Mapping[str, Any]:
        inputs = {k: v for k, v in inputs.items()}

        # Turn input ids into 1-hots or embeddings
        inputs_embeds = self._embed(inputs["input_ids"])

        # Scatter parameters into inputs using input map
        for name, attack in self.params.items():
            # Check if attack exists in inputs
            name = MASK_FORMAT % name
            if name not in inputs:
                continue

            inputs_masks: torch.BoolTensor = inputs[name]  # type: ignore
            inputs_embeds = self._scatter_attack(attack, inputs_masks, inputs_embeds)

        # Project adv inputs to embeddings and ids
        if self.dim == 0:
            input_ids = inputs_embeds.argmax(-1)
            inputs_embeds = torch.matmul(inputs_embeds, self.embedding.weight)
        else:
            input_ids = torch.cdist(
                inputs_embeds.detach(), self.embedding.weight
            ).argmin(-1)

        inputs["input_ids"] = input_ids
        inputs["inputs_embeds"] = inputs_embeds
        return inputs

    def _scatter_attack(
        self,
        attack: torch.Tensor,
        inputs_masks: torch.BoolTensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        outputs = []
        for input_embeds, input_mask in zip(inputs_embeds, inputs_masks):
            # find tokens corresponding to attack using map
            input_index = torch.where(input_mask)[0]

            # if there is no attack in this input, just skip it
            if len(input_index) == 0:
                outputs.append(input_embeds)
                continue

            # repeat index along ultimate dimension
            input_index = input_index[..., None].repeat(1, input_embeds.shape[-1])

            # tile attack to match repeats
            attack = attack.tile(len(input_index) // len(attack), 1)

            if len(input_index) != len(attack):
                raise ValueError(
                    f"Attack is {len(attack)} tokens but found {len(input_index)} in the index!"
                )

            # Scatter attack into input using index
            input_embeds = input_embeds.scatter(dim=0, index=input_index, src=attack)
            outputs.append(input_embeds)

        outputs = torch.stack(outputs)
        return outputs

    def _fast_forward(self, inputs: MutableMapping[str, Any]) -> Mapping[str, Any]:
        inputs = {k: v for k, v in inputs.items()}
        adv_input_ids = inputs.pop("input_ids")

        # Scatter parameters into inputs using input map
        for name, attack in self.params.items():
            # Check if attack exists in inputs
            name = MASK_FORMAT % name
            if name not in inputs:
                continue

            inputs_masks: torch.BoolTensor = inputs[name]  # type: ignore
            adv_input_ids = self._fast_scatter_attack(
                attack, inputs_masks, adv_input_ids
            )

        inputs["input_ids"] = adv_input_ids
        return inputs

    def _fast_scatter_attack(
        self,
        attack: torch.Tensor,
        inputs_masks: torch.BoolTensor,
        inputs_ids: torch.Tensor,
    ) -> torch.Tensor:
        attack = attack.argmax(-1)
        outputs = []
        for input_ids, input_mask in zip(inputs_ids, inputs_masks):
            # find tokens corresponding to attack using map
            input_index = torch.where(input_mask)[0]

            # if there is no attack in this input, just skip it
            if len(input_index) == 0:
                outputs.append(input_ids)
                continue

            # tile attack to match repeats
            attack = attack.tile(len(input_index) // len(attack))

            if len(input_index) != len(attack):
                raise ValueError(
                    f"Attack is {len(attack)} tokens but found {len(input_index)} in the index!"
                )

            # Scatter attack into input using index
            input_ids = input_ids.scatter(dim=0, index=input_index, src=attack)
            outputs.append(input_ids)

        outputs = torch.stack(outputs)
        return outputs
