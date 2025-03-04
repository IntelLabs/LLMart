#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
# type: ignore
#
# Makefile for random strings example

PYTHON=python3.11
VENV_DIR=venv

STRING=$(shell tr -dc 'A-Za-z0-9!#%&'\''()*+,-./:;<=>?@[\]^_{|}~' </dev/urandom | head -c 10; echo)
SCRIPT=whitebox.py
MAX_STEPS=2
LR=0.005

ARGS=--max_steps $(MAX_STEPS)  --lr $(LR)

all: install run

install:
	@if [ ! -d $(VENV_DIR) ]; then \
			$(PYTHON) -m venv $(VENV_DIR); \
	fi
	. $(VENV_DIR)/bin/activate && \
	cd ../.. && \
	pip install -e ".[core,dev]";

run:
	$(VENV_DIR)/bin/$(PYTHON) $(SCRIPT) $(ARGS) "$(STRING)";

clean:
	rm -rf __pycache__ $(VENV_DIR)

.PHONY: all install run clean