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
PKG_MGR=uv

string=$(shell tr -dc 'A-Za-z0-9!#%&'\''()*+,-./:;<=>?@[\]^_{|}~' </dev/urandom | head -c 10; echo)
script=whitebox.py
max_steps=2
lr=0.005

args=--max_steps $(max_steps)  --lr $(lr)

all: install run

install:
	@if [ ! -d ../../$(VENV_DIR) ]; then \
		cd ../.. && $(PKG_MGR) venv $(VENV_DIR); \
	fi
	. ../../$(VENV_DIR)/bin/activate && \
	cd ../.. && \
	$(PKG_MGR) pip install -e".[core,dev]"

run:
	. ../../$(VENV_DIR)/bin/activate && $(PYTHON) $(script) $(args) "$(string)";

clean:
	-rm -rf __pycache__ ../../$(VENV_DIR)

.PHONY: all install run clean
