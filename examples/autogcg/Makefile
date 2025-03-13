#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

include ../preamble.mk

time_budget_s=120
subset=2
steps=2
num_seeds=2
args=--subset $(subset) --time_budget_s $(time_budget_s) --steps $(steps) --num_seeds $(num_seeds)

all: run

run:
> uv run --with-requirements requirements.txt main.py $(args)

.PHONY: all run
