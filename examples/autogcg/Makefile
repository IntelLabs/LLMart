#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
# type: ignore
#
# Makefile for autoGCG example

PYTHON=python3.11
VENV_DIR=venv

subset=2
steps=2
num_gpu=4
script=main.py
args=--subset $(subset) --steps $(steps)
mode=$(if $(gpu),gpu,cpu) # mode is 'gpu' if GPU=1, otherwise 'cpu'

all: install run

install:
	@if [ ! -d $(VENV_DIR) ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	. $(VENV_DIR)/bin/activate && \
	pip install -r requirements.txt && \
	cd ../.. && \
	pip install -e".[core,dev]"

run:
	@if [ $(mode) = "gpu" ]; then \
		ts -nfG$(num_gpu) $(VENV_DIR)/bin/$(PYTHON) $(script) $(args); \
	elif [ $(mode) = "cpu" ]; then \
		$(VENV_DIR)/bin/$(PYTHON) $(script) $(args); \
	else \
		echo "Invalid mode. Aborting."; exit 1; \
	fi

clean:
	rm -rf __pycache__ .installed $(VENV_DIR)

.PHONY: all install run clean