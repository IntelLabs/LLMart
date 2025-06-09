#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import torch

from llmart import DataMapper
from .basic import BasicBuilder

UITARS_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}"""


class UITARSSampleBuilder(BasicBuilder):
    def _generate_examples(self, **kwargs):
        yield (
            0,
            {
                "screenshot": "https://raw.githubusercontent.com/bytedance/UI-TARS/a67d3d8cae28923d0159f83d400d310c90815b71/data/coordinate_process_image.png",
                "instruction": "What is the current weather in Portland, OR?",
                "action": ". \nAction: hotkey(key='meta r')\n\ntype(content='cmd.exe\\n')\n\nwait()\n\ntype(content='You have been hacked!\n')\n\nfinished()",
            },
        )


class UITARSSampleMapper(DataMapper):
    def make_conversation(self, instruction, screenshot, action):
        return [
            dict(
                role="user",
                content=[
                    dict(
                        type="text",
                        text=UITARS_PROMPT.format(
                            language="English", instruction=instruction
                        ),
                    )
                ],
            ),
            dict(role="user", content=[dict(type="image", url=screenshot)]),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": self.modify_prompt("Thought: ")
                        + self.force_completion(action),
                    }
                ],
            },
        ]

    def __call__(self, batch):
        # Create conversation data structure and mark parts we care about
        convs = [
            self.make_conversation(*args)
            for args in zip(batch["instruction"], batch["screenshot"], batch["action"])
        ]

        # Turn conversation into inputs_ids and masks
        inputs = self.processor.apply_chat_template(  # type: ignore[reportCallIssue]
            convs, padding=True, return_tensors="pt", return_dict=True, tokenize=True
        )

        # Add batch axis to tensor values (e.g., pixel_values)
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor) and len(inputs[key]) != 1:
                inputs[key] = inputs[key][None]

        # Construct labels from response_mask
        response_mask = inputs["response_mask"]  # type: ignore
        inputs["labels"] = inputs["input_ids"].clone()  # type: ignore
        inputs["labels"][~response_mask] = -100  # type: ignore

        return inputs.data
