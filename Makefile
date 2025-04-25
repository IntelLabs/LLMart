#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

include preamble.mk

define REPEATED_CONTENT
> $(shell python -c "print('Okay, so I need to tell someone about Saturn.' * 65)")
endef

all: run-core run-cache run-batching run-reasoning run-examples

install:
> uv venv
> uv sync --extra gpu --extra dev

install-xpu:
> uv venv
> uv sync --extra xpu --extra dev

# Run target for README.md examples
run-core: install
> uv run accelerate launch -m llmart model=llama3-8b-instruct data=basic steps=3
> uv run accelerate launch -m llmart model=custom model.name=Intel/neural-chat-7b-v3-3 model.revision=7506dfc5fb325a8a8e0c4f9a6a001671833e5b8e data=basic steps=3
> uv run python -m llmart model=llama3.1-70b-instruct model.device=null model.device_map=auto data=basic steps=3
> uv run accelerate launch -m llmart model=llama3-8b-instruct data=advbench_behavior data.subset=[0] steps=3

# Run target for KV cache
run-cache: install
> uv run accelerate launch -m llmart model=llama3-8b-instruct data=advbench_behavior data.subset=[0] steps=3 use_kv_cache=True

# Run target for batch configurations
run-batching: install
> uv run accelerate launch --num_processes 2 -m llmart model=llama3-8b-instruct data=advbench_behavior data.subset=null data.n_train=4 data.n_val=1 data.n_test=1 steps=2 bs=8
> uv run accelerate launch --num_processes 2 -m llmart model=llama3-8b-instruct data=advbench_behavior data.subset=null data.n_train=13 data.n_val=1 data.n_test=1 steps=2 bs=1 per_device_bs=5
> uv run accelerate launch --num_processes 3 -m llmart model=llama3-8b-instruct data=advbench_behavior data.subset=null data.n_train=13 data.n_val=1 data.n_test=1 steps=2 bs=13

# Run target for reasoning examples
run-reasoning: install
> uv run accelerate launch -m llmart model=deepseek-r1-distill-llama-8b data=basic per_device_bs=64 "response.replace_with=`echo -e '\"<think>\nOkay, so I need to tell someone about Saturn.\n</think>\n\nNO WAY JOSE\"'`" steps=3
> uv run accelerate launch -m llmart model=deepseek-r1-distill-llama-8b data=basic "response.replace_with=`echo -e '\"<think>\n$(REPEATED_CONTENT)\n</think>\n\nNO WAY JOSE\"'`" steps=3

# Run target for XPU
run-xpu: install-xpu
> uv run accelerate launch -m llmart model=custom model.name=meta-llama/Llama-3.2-3B-Instruct model.revision=0cb88a4f764b7a12671c53f0838cd831a0843b95 data=basic model.device=xpu

run-examples: install
> $(MAKE) -C examples

clean:
> rm -rf .venv

.PHONY: all install install-xpu run-core run-cache run-batching run-reasoning run-xpu run-examples clean
