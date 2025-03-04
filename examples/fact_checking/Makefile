#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
# type: ignore
#
# Makefile for fact_checking example

PYTHON=python3.11
VENV_DIR=.venv

NUM_STEPS=2
NUM_GPU=4
SCRIPT=claim.py # Can be either 'claim.py' or 'document.py'
ARGS=--num_steps $(NUM_STEPS)

# Because the scripts will verify the optimized suffix using the reference vllm pipeline, these examples require at least two GPUs (or a single GPU with at least 80 GB of VRAM).
MODE=gpu

all: install run

install:
	@if [ ! -d $(VENV_DIR) ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	. $(VENV_DIR)/bin/activate && \
	pip install -r requirements.txt && \
	cd ../.. && \
	pip install -e".[core,dev]" && \
	pip install nltk && \
	$(PYTHON) -c "import nltk; nltk.download('punkt')"

run:
	@if [ $(MODE) = "gpu" ]; then \
		ts -nfG$(NUM_GPU) $(VENV_DIR)/bin/$(PYTHON) $(SCRIPT) $(ARGS); \
	elif [ $(MODE) = "cpu" ]; then \
		echo "Cannot run in CPU mode. Aborting."; exit 1; \
	else \
		echo "Invalid mode. Aborting."; exit 1; \
	fi

clean:
	rm -rf __pycache__ $(VENV_DIR)

.PHONY: all install run clean
