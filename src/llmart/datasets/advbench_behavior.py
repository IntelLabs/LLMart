#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

from datasets import load_dataset


def get_conversations(data_files: str) -> list[list[dict[str, str]]]:
    # Read the CSV file
    data = load_dataset("csv", data_files=data_files, cache_dir=".")

    # Convert to conversation format
    conversations = []
    for target, goal in zip(data["train"]["target"], data["train"]["goal"]):  # type: ignore
        conversation_pair = [
            dict(role="user", content=goal),
            dict(role="assistant", content=target),
        ]
        conversations.append(conversation_pair)

    return conversations
