#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

include ../../preamble.mk

prompt=Who is Harry Potter?
completion=Harry Potter is the main protagonist in J.K. Rowling's series of fantasy novels

max_steps=2
max_optim_tokens=32
lr=0.005
args=--max_steps $(max_steps) --max_optim_tokens $(max_optim_tokens) --lr $(lr)

all: run

run:
> uv run --with-requirements requirements.txt whitebox.py "$(prompt)" "$(completion)" $(args)

.PHONY: all run
