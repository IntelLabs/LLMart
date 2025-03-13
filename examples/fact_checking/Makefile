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
> cd examples/fact_checking && \
> uv pip install nltk && \
> uv pip install -r requirements.txt && \
> python -c "import nltk; nltk.download('punkt')"

run:
> source ../../$(VENV_DIR)/bin/activate && \
> python claim.py $(args) && \
> python document.py $(args)

clean:
> rm -rf __pycache__ ../../$(VENV_DIR)

.PHONY: all install run clean
