#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

include preamble.mk

all: install run-core run-examples clean

install:
> uv venv
> source .venv/bin/activate
> uv sync --all-extras

# Run target for README.md examples
run-core:
> source .venv/bin/activate
> accelerate launch -m llmart model=llama3-8b-instruct data=basic loss=model steps=3
> accelerate launch -m llmart model=custom model.name=Intel/neural-chat-7b-v3-3 model.revision=7506dfc5fb325a8a8e0c4f9a6a001671833e5b8e data=basic loss=model steps=3
> accelerate launch -m llmart model=deepseek-r1-distill-llama-8b data=basic per_device_bs=64 "response.replace_with=`echo -e '\"<think>\nOkay, so I need to tell someone about Saturn.\n</think>\n\nNO WAY JOSE\"'`" steps=3
> python -m llmart model=llama3.1-70b-instruct model.device=null model.device_map=auto data=basic loss=model steps=3
> accelerate launch -m llmart model=llama3-8b-instruct data=advbench_behavior data.subset=[0] loss=model steps=3

run-examples:
> source .venv/bin/activate
> $(MAKE) -C examples

clean:
> rm -rf .venv

.PHONY: all install run-core run-examples clean
