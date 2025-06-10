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
> wget https://raw.githubusercontent.com/ML-GSAI/LLaDA/f51cb1731f5a40ba35c15e51b6b66b147e689f24/generate.py
> $(RUN_GPU) --with-requirements requirements.txt main.py $(args)

.PHONY: all run
