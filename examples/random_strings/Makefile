#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

include ../preamble.mk

string=$(shell tr -dc 'A-Za-z0-9!#%&'\''()*+,-./:;<=>?@[\]^_{|}~' </dev/urandom | head -c 10; echo)

max_steps=2
lr=0.005
args=--max_steps $(max_steps) --lr $(lr)

all: run

run:
> uv run whitebox.py $(args) "$(string)"

.PHONY: all run
