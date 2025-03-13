#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#


VENV_DIR=.venv

time_budget_s=120
subset=2
steps=2
num_seeds=2
script=main.py
args=--subset $(subset) --time_budget_s $(time_budget_s) --steps $(steps) --num_seeds $(num_seeds)

all: install run clean

install:
	cd ../.. && uv venv $(VENV_DIR) && uv sync --all-extras && \
	. $(VENV_DIR)/bin/activate && \
	cd examples/autogcg && \
	uv pip install -r requirements.txt

run:
	. ../../$(VENV_DIR)/bin/activate && \
	python main.py $(args)

clean:
	rm -rf __pycache__ ../../$(VENV_DIR)

.PHONY: all install run clean
