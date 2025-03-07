#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
# type: ignore

# Makefile for unlearning example

PYTHON=python3.11
VENV_DIR=venv

max_steps=2
max_optim_tokens=32
lr=0.005
num_gpu=4
script=whitebox.py

prompt=Who is Harry Potter?
completion=Harry Potter is the main protagonist in J.K. Rowling's series of fantasy novels
args=--max_steps $(max_steps) --max_optim_tokens $(max_optim_tokens) --lr $(lr)
mode=$(if $(gpu),gpu,cpu) # mode is 'gpu' if gpu=1, otherwise 'cpu'

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
        args="$(args) --use_hard_tokens"; \
    fi; \
    if [ $(mode) = "gpu" ]; then \
        echo "Running with $(num_gpu) GPUs"; \
        ts -nfG$(num_gpu) $(VENV_DIR)/bin/$(PYTHON) $(script) "$(prompt)" "$(completion)" $(args); \
    elif [ $(mode) = "cpu" ]; then \
        echo "Running with CPU"; \
        $(VENV_DIR)/bin/$(PYTHON) $(script) "$(prompt)" "$(completion)" $(args); \
    else \
        echo "Invalid mode. Aborting."; exit 1; \
    fi

clean:
    rm -rf __pycache__ .installed $(VENV_DIR)

.PHONY: all install run clean