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
> uv run --with-requirements requirements.txt main.py $(args)

.PHONY: all run
