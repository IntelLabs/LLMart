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

SUBSET=2
STEPS=2
NUM_GPU=4
SCRIPT=main.py
ARGS=--subset $(SUBSET) --steps $(STEPS)
MODE=$(if $(GPU),gpu,cpu) # MODE is 'gpu' if GPU=1, otherwise 'cpu'

all: install run

install: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate:
	@if [ ! -d $(VENV_DIR) ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	. $(VENV_DIR)/bin/activate && \
	pip install -r requirements.txt && \
	cd ../.. && \
	pip install -e".[core,dev]"

run:
	@if [ $(MODE) = "gpu" ]; then \
		ts -nfG$(NUM_GPU) $(VENV_DIR)/bin/$(PYTHON) $(SCRIPT) $(ARGS); \
	elif [ $(MODE) = "cpu" ]; then \
		$(VENV_DIR)/bin/$(PYTHON) $(SCRIPT) $(ARGS); \
	else \
		echo "Invalid mode. Aborting."; exit 1; \
	fi

clean:
	rm -rf __pycache__ .installed $(VENV_DIR)

.PHONY: all install run clean