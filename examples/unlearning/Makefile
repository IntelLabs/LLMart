#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

include ../preamble.mk

VENV_DIR=.venv

prompt=Who is Harry Potter?
completion=Harry Potter is the main protagonist in J.K. Rowling's series of fantasy novels

max_steps=2
max_optim_tokens=32
lr=0.005
args=--max_steps $(max_steps) --max_optim_tokens $(max_optim_tokens) --lr $(lr)

all: install run clean

install:
> cd ../.. && uv venv $(VENV_DIR) && uv sync --all-extras && \
> source $(VENV_DIR)/bin/activate && \
> cd examples/unlearning && \
> uv pip install -r requirements.txt

run:
> source ../../$(VENV_DIR)/bin/activate && \
> python whitebox.py "$(prompt)" "$(completion)" $(args)

clean:
> rm -rf __pycache__ ../../$(VENV_DIR)

.PHONY: all install run clean
