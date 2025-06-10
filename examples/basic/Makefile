#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

include ../../preamble.mk

num_steps=2
args=--num_steps $(num_steps)

all: run

run:
> $(RUN_GPU) --with-requirements requirements.txt main.py $(args)

run_notebook:
> $(RUN_GPU) --with-requirements requirements.txt jupyter execute basic_dev_workflow.ipynb

.PHONY: all run run_notebook
