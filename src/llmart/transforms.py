#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import re
import torch
from copy import deepcopy
from typing import TypeAlias, overload
from abc import ABC, abstractmethod

from .config import ResponseConf, AttackConf
from .tokenizer import BEGIN_TAG_FORMAT, END_TAG_FORMAT

Conversation: TypeAlias = list[dict[str, str]]
Elements: TypeAlias = dict[str, tuple[str, str, str]]


class Transform(torch.nn.Module, ABC):
    """Base class for sequence or conversation transformations.

    Args:
        conv_role: Role in conversation to transform ('user', 'assistant', 'system')
        default_token: Default token to use to create attack sequences
    """

    def __init__(self, conv_role, default_token: str = " !"):
        super().__init__()

        self.conv_role = conv_role
        self.default_token = default_token
        self._elements: Elements = {}

    def _wrap(self, **name_content: str | int) -> str:
        assert len(name_content) == 1
        name, content = name_content.popitem()

        if isinstance(content, int):
            content = self.default_token * content

        begin_tag = BEGIN_TAG_FORMAT % name
        end_tag = END_TAG_FORMAT % name

        self._elements[name] = (begin_tag, content, end_tag)

        return begin_tag + content + end_tag

    @property
    def tags(self) -> list[str]:
        tags: list[str] = []
        for begin_tag, _, end_tag in self._elements.values():
            tags.extend([begin_tag, end_tag])
        return tags

    @property
    def elements(self) -> list[str]:
        return [
            f"{begin_tag}{content}{end_tag}"
            for begin_tag, content, end_tag in self._elements.values()
        ]

    def __str__(self) -> str:
        return "".join(self.elements)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)})"

    @abstractmethod
    def transform(self, conv: str) -> str:
        raise NotImplementedError

    @overload
    def forward(self, conv: str) -> str: ...

    @overload
    def forward(self, conv: Conversation) -> Conversation: ...

    def forward(self, conv):
        if isinstance(conv, str):
            return self.transform(conv)

        else:
            # Copy because we modify in-place below
            conv = deepcopy(conv)

            # Add attack prefix to each user's content
            for turn in conv:
                if turn["role"] == self.conv_role:
                    turn["content"] = self.transform(turn["content"])

            return conv


class AttackPrompt(Transform):
    def __init__(
        self,
        prefix: str | int | None = None,
        suffix: str | int | None = None,
        repl: str | int | None = None,
        pattern: str | None = None,
        default_token: str = " !",
        prefix_pad_left: str = "",
        prefix_pad_right: str = " ",
        suffix_pad_left: str = " ",
        suffix_pad_right: str = "",
        repl_pad_left: str = "",
        repl_pad_right: str = "",
    ):
        """Adds prefix/suffix tags or pattern replacements to user conversation messages.

        Args:
            prefix: Content to prepend (str or token count)
            suffix: Content to append (str or token count)
            repl: Replacement pattern content
            pattern: Regex pattern to match for replacement
            default_token: Token to use for integer prefix/suffix
            prefix_pad_left: Padding before prefix
            prefix_pad_right: Padding after prefix
            suffix_pad_left: Padding before suffix
            suffix_pad_right: Padding after suffix
            repl_pad_left: Padding before replacement
            repl_pad_right: Padding after replacement
        """

        super().__init__(conv_role="user", default_token=default_token)
        self.pattern = pattern

        # create amortized attacks elements
        self.prefix = (
            prefix_pad_left + self._wrap(prefix=prefix) + prefix_pad_right
            if prefix
            else ""
        )
        self.suffix = (
            suffix_pad_left + self._wrap(suffix=suffix) + suffix_pad_right
            if suffix
            else ""
        )
        self.repl = (
            repl_pad_left + self._wrap(repl=repl) + repl_pad_right if repl else None
        )

    def transform(self, conv: str) -> str:
        if self.pattern and self.repl:
            conv = re.sub(self.pattern, self.repl.replace("\\", r"\\"), conv)
        return self.prefix + conv + self.suffix


class MaskCompletion(Transform):
    """Wraps assistant responses in special response tags.

    Args:
        replace_with: Optional text to replace response content
        prefix: Optional text to prepend to the response content
        suffix: Optional text to append to the response content
    """

    def __init__(
        self,
        replace_with: str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
    ):
        super().__init__(conv_role="assistant")
        self.replace_with = replace_with
        self.prefix = prefix or ""
        self.suffix = suffix or ""

        # create amortize response tag
        self._wrap(response="")

    def transform(self, conv: str) -> str:
        return self._wrap(
            response=self.prefix + (self.replace_with or conv) + self.suffix
        )


def from_config(cfg: ResponseConf | AttackConf) -> Transform:
    """Create Transform instance from configuration.

    Args:
        cfg: Response or Attack configuration object

    Returns:
        Configured Transform instance
    """

    if isinstance(cfg, ResponseConf):
        return MaskCompletion(cfg.replace_with, cfg.prefix, cfg.suffix)

    if isinstance(cfg, AttackConf):
        return AttackPrompt(
            cfg.prefix,
            cfg.suffix,
            cfg.repl,
            cfg.pattern,
            cfg.default_token,
            cfg.prefix_pad_left,
            cfg.prefix_pad_right,
            cfg.suffix_pad_left,
            cfg.suffix_pad_right,
            cfg.repl_pad_left,
            cfg.repl_pad_right,
        )
