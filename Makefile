#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

include preamble.mk

define REPEATED_CONTENT
> $(shell python -c "print('Okay, so I need to tell someone about Saturn.' * 65)")
endef

all: run-core run-cache run-batching run-reasoning run-examples run-image-text-to-text run-dataset

# Run target for README.md examples
run-core:
> $(RUN_GPU) accelerate launch -m llmart model=llama3-8b-instruct data=basic steps=3
> $(RUN_GPU) accelerate launch -m llmart model=custom model.name=Intel/neural-chat-7b-v3-3 model.revision=7506dfc5fb325a8a8e0c4f9a6a001671833e5b8e data=basic steps=3
> $(RUN_GPU) python -m llmart model=llama3.1-70b-instruct model.device=null model.device_map=auto data=basic steps=3
> $(RUN_GPU) accelerate launch -m llmart model=llama3-8b-instruct data=advbench_behavior data.subset=[0] steps=3

# Run target for KV cache
run-cache:
> $(RUN_GPU) accelerate launch -m llmart model=llama3-8b-instruct data=advbench_behavior data.subset=[0] steps=3 use_kv_cache=True

# Run target for batch configurations
run-batching:
> $(RUN_GPU) accelerate launch --num_processes 2 -m llmart model=llama3-8b-instruct data=advbench_behavior data.subset=null data.n_train=4 data.n_val=1 data.n_test=1 steps=2 bs=8
> $(RUN_GPU) accelerate launch --num_processes 2 -m llmart model=llama3-8b-instruct data=advbench_behavior data.subset=null data.n_train=13 data.n_val=1 data.n_test=1 steps=2 bs=1 per_device_bs=5
> $(RUN_GPU) accelerate launch --num_processes 3 -m llmart model=llama3-8b-instruct data=advbench_behavior data.subset=null data.n_train=13 data.n_val=1 data.n_test=1 steps=2 bs=13

# Run target for reasoning examples
run-reasoning:
> $(RUN_GPU) accelerate launch -m llmart model=deepseek-r1-distill-llama-8b data=basic per_device_bs=64 "response.prefix=`echo -e '\"<think>\nOkay, so I need to tell someone about Saturn.\n</think>\n\n\"'`" steps=3
> $(RUN_GPU) accelerate launch -m llmart model=deepseek-r1-distill-llama-8b data=basic "response.prefix=`echo -e '\"<think>\n$(REPEATED_CONTENT)\n</think>\n\n\"'`" steps=3

# Run target for XPU
run-xpu:
> $(RUN_XPU) accelerate launch -m llmart model=custom model.name=meta-llama/Llama-3.2-3B-Instruct model.revision=0cb88a4f764b7a12671c53f0838cd831a0843b95 data=basic model.device=xpu

# Run image-text-to-text examples
run-image-text-to-text:
> $(RUN_GPU) accelerate launch -m llmart model=ui-tars-2b-sft data=ui-tars attack.suffix=20 attack.suffix_pad_left="" per_device_bs=1 optim.n_swaps=16 steps=3
# NOTE: The example below does not run because mllama does not allow feeding pixel_values and inputs_embeds
#> $(RUN_GPU) accelerate launch -m llmart model=llama3.2-11b-vision-instruct data=mllama per_device_bs=16 attack.suffix=40 steps=3

# Run dataset examples
run-dataset:
> $(RUN_GPU) accelerate launch -m llmart model=harmbench-classifier data=harmbench 'data.subset=[2]' per_device_bs=8 steps=3
> $(RUN_GPU) accelerate launch -m llmart model=llamaguard3-1b data=toxic-chat 'data.subset=[0]' attack.suffix_pad_right="." data.mapper=toxic_chat_model_output steps=3

run-examples:
> $(MAKE) -C examples

clean:
> rm -rf .venv

.PHONY: all run-core run-cache run-batching run-reasoning run-xpu run-image-text-to-text run-dataset run-examples clean
