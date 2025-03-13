#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

include ../preamble.mk

VENV_DIR=.venv

string=$(shell tr -dc 'A-Za-z0-9!#%&'\''()*+,-./:;<=>?@[\]^_{|}~' </dev/urandom | head -c 10; echo)

max_steps=2
lr=0.005
args=--max_steps $(max_steps) --lr $(lr)

all: install run clean

install:
> cd ../.. && uv venv $(VENV_DIR) && uv sync --all-extras && \
> source $(VENV_DIR)/bin/activate && \
> cd examples/random_strings \

run:
> source ../../$(VENV_DIR)/bin/activate && \
> python whitebox.py $(args) "$(string)"

clean:
> rm -rf __pycache__ ../../$(VENV_DIR)

.PHONY: all install run clean
