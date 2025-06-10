#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#


from llmart import DataMapper


class ToxicChatUserInputMapper(DataMapper):
    def make_conversation(self, user_input, toxicity):
        # NOTE: We flip labels here because we want to attack the model!
        assessment = "unsafe" if toxicity == 0 else "safe"
        return [
            dict(
                role="user",
                content=[dict(type="text", text=self.modify_prompt(user_input))],
                assessment=self.force_completion(assessment),
            ),
        ]

    def __call__(self, batch):
        # Turn batch into conversation suitable for apply_chat_template
        return dict(
            conversation=[
                self.make_conversation(*args)
                for args in zip(batch["user_input"], batch["toxicity"])
            ]
        )
