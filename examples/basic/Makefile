#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

include ../preamble.mk

VENV_DIR=.venv

num_steps=2
args=--num_steps $(num_steps)

all: install run clean

install:
> cd ../.. && uv venv $(VENV_DIR) && uv sync --all-extras && \
> source $(VENV_DIR)/bin/activate && \
> cd examples/basic && \
> uv pip install -r requirements.txt

run:
> source ../../$(VENV_DIR)/bin/activate && \
> python main.py $(args)

clean:
> rm -rf __pycache__ ../../$(VENV_DIR)

.PHONY: all install run clean
