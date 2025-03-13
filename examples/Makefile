#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

include preamble.mk

EXAMPLE_DIRS := basic autogcg fact_checking llmguard random_strings unlearning

all: install run clean

install:
> cd ..
> uv venv
> source .venv/bin/activate
> uv sync --all-extras
> cd examples

# Generate a run target that depends on all run_X targets
run:
> $(foreach dir,$(EXAMPLE_DIRS),make -C $(dir) &&) true

clean:
> rm -rf .venv

.PHONY: all install run clean
