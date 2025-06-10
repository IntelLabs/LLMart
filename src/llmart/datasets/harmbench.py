#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

from datasets import load_dataset, Dataset

from llmart import DataMapper


class HarmBenchMapper(DataMapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        behaviors = load_dataset(  # type: ignore[reportArgumentType]
            "csv",
            data_files="https://raw.githubusercontent.com/centerforaisafety/HarmBench/refs/heads/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
            split="train",
        )
        assert isinstance(behaviors, Dataset)
        self.behaviors = dict(zip(behaviors["BehaviorID"], behaviors["Behavior"]))

    def make_conversation(
        self, behavior_id, generation, human_0, human_1, human_2, **sample
    ):
        # NOTE: We flip the human classification because we attack!
        assessment = "No" if human_0 + human_1 + human_2 >= 2 else "Yes"
        return [
            dict(
                behavior=self.behaviors[behavior_id],
                generation=self.modify_prompt(generation),
                assessment=self.force_completion(assessment),
            ),
        ]

    def __call__(self, batch):
        return dict(
            conversation=[
                self.make_conversation(**sample)
                for samples in batch.values()
                for sample in samples
                if sample["behavior_id"] in self.behaviors
            ]
        )
