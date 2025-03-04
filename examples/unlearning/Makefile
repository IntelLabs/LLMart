#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
# type: ignore

# Makefile for unlearning example

PYTHON=python3.11
VENV_DIR=venv

MAX_STEPS=2
MAX_OPTIM_TOKENS=32
LR=0.005
NUM_GPU=4
SCRIPT=whitebox.py

PROMPT=Who is Harry Potter?
COMPLETION=Harry Potter is the main protagonist in J.K. Rowling's series of fantasy novels
ARGS=--max_steps $(MAX_STEPS) --max_optim_tokens $(MAX_OPTIM_TOKENS) --lr $(LR)
MODE=$(if $(GPU),gpu,cpu) # MODE is 'gpu' if GPU=1, otherwise 'cpu'

all: install run

install:
    @if [ ! -d $(VENV_DIR) ]; then \
        $(PYTHON) -m venv $(VENV_DIR); \
    fi
    . $(VENV_DIR)/bin/activate && \
    pip install -r requirements.txt && \
    cd ../.. && \
    pip install -e ".[core,dev]"

run:
    @if [ "$(findstring use_hard_tokens, $(MAKEFLAGS))" ]; then \
        echo "Using hard tokens"; \
        ARGS="$(ARGS) --use_hard_tokens"; \
    fi; \
    if [ $(MODE) = "gpu" ]; then \
        echo "Running with $(NUM_GPU) GPUs"; \
        ts -nfG$(NUM_GPU) $(VENV_DIR)/bin/$(PYTHON) $(SCRIPT) "$(PROMPT)" "$(COMPLETION)" $(ARGS); \
    elif [ $(MODE) = "cpu" ]; then \
        echo "Running with CPU"; \
        $(VENV_DIR)/bin/$(PYTHON) $(SCRIPT) "$(PROMPT)" "$(COMPLETION)" $(ARGS); \
    else \
        echo "Invalid mode. Aborting."; exit 1; \
    fi

clean:
    rm -rf __pycache__ .installed $(VENV_DIR)

.PHONY: all install run clean