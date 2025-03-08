#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
# type: ignore
#
# This Makefile is used to install dependencies for LLMart and run the tests required for release and testing.

PYTHON=python3.11
VENV_DIR=venv
PKG_MGR=uv
NUM_GPU=4

all: run

create-env:
	@if [ ! -d $(VENV_DIR) ]; then \
		echo "Creating virtual environment..."; \
		cd .. && $(PKG_MGR) venv $(VENV_DIR); \
	else \
		echo "Virtual environment already exists."; \
	fi

install: create-env
	. ../$(VENV_DIR)/bin/activate && \
	cd .. && $(PKG_MGR) pip install -e ".[core,dev]"

run: install
	@if echo "$(ARGS)" | grep -qE "model\.device=cuda(\s|$$)"; then \
		cd .. && . $(VENV_DIR)/bin/activate && ts -nfG$(NUM_GPU) accelerate launch -m llmart $(ARGS); \
	elif echo "$(ARGS)" | grep -qE "model\.device=cpu(\s|$$)"; then \
		cd .. && . $(VENV_DIR)/bin/activate && accelerate launch -m llmart $(ARGS); \
	else \
		echo "Invalid DEVICE option. Please use cuda or cpu. Aborting."; exit 1; \
	fi

clean:
	rm -rf __pycache__ $(VENV_DIR)

.PHONY: all create-env install run clean