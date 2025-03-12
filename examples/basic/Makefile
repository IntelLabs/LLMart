#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

PYTHON=python3.11
VENV_DIR=venv
PKG_MGR=uv

num_steps=2
num_gpu=4
script=main.py
args=--num_steps $(num_steps)
mode=$(if $(gpu),gpu,cpu) # mode is 'gpu' if gpu=1, otherwise 'cpu'

all: install run

install:
	@if [ ! -d ../../$(VENV_DIR) ]; then \
		cd ../.. && $(PKG_MGR) venv $(VENV_DIR); \
	fi
	. ../../$(VENV_DIR)/bin/activate && \
	$(PKG_MGR) pip install -r requirements.txt && \
	cd ../.. && \
	$(PKG_MGR) pip install -e".[core,dev]"

run:
	@if [ $(mode) = "gpu" ]; then \
		. ../../$(VENV_DIR)/bin/activate && ts -nfG$(num_gpu) $(PYTHON) $(script) $(args); \
	elif [ $(mode) = "cpu" ]; then \
		. ../../$(VENV_DIR)/bin/activate && $(PYTHON) $(script) $(args); \
	else \
		echo "Invalid mode. Aborting."; exit 1; \
	fi

clean:
	-rm -rf __pycache__ .installed ../../$(VENV_DIR)

.PHONY: all install run clean
