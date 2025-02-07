#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

from llmart import Transform
from .adv_csv import AdvCsv


class AdvBenchBehavior(AdvCsv):
    def to_conversations(self, ds) -> list[list[dict]]:
        mark_prompt: Transform = self.config.mark_prompt  # type: ignore
        mark_completion: Transform = self.config.mark_completion  # type: ignore
        # Turn batch into conversation suitable for apply_chat_template
        convs = []
        for d in ds:
            assert isinstance(d, dict)
            conv = [
                dict(role="user", content=mark_prompt(d["goal"])),
                dict(role="assistant", content=mark_completion(d["target"])),
            ]
            convs.append(conv)
        return convs
