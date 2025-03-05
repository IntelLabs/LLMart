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

MODEL=llama3-8b-instruct
DATA=basic
LOSS=model
STEPS=2
PER_DEVICE_BS=1000
DEVICE=cuda
ARGS=model=$(MODEL) model.device=$(DEVICE) data=$(DATA) loss=$(LOSS) steps=$(STEPS) per_device_bs=$(PER_DEVICE_BS)

all: install run

create-env:
	@if [ ! -d $(VENV_DIR) ]; then \
		echo "Creating virtual environment..."; \
		$(PKG_MGR) venv $(VENV_DIR); \
	else \
		echo "Virtual environment already exists."; \
	fi

install: create-env
	. $(VENV_DIR)/bin/activate && \
	$(PKG_MGR) pip install -e ".[core,dev]"

run:
	. $(VENV_DIR)/bin/activate && \
	accelerate launch -m llmart $(ARGS)

run:
	. $(VENV_DIR)/bin/activate && \
	accelerate launch -m llmart $(ARGS)

clean:
	rm -rf __pycache__ $(VENV_DIR)

.PHONY: all create-env install run clean