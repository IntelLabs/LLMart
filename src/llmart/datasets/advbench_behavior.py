#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

from llmart import DataMapper


class AdvBenchBehaviorMapper(DataMapper):
    def __call__(self, batch):
        convs = [
            [
                dict(role="user", content=self.modify_prompt(goal)),
                dict(role="assistant", content=self.force_completion(target)),
            ]
            for goal, target in zip(batch["goal"], batch["target"])
        ]
        return dict(conversation=convs)
