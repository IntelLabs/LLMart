#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

include ../preamble.mk

EXAMPLE_DIRS := basic autogcg fact_checking llmguard random_strings unlearning

all: run

# Run target that iterates over all examples
run:
> $(foreach dir,$(EXAMPLE_DIRS),$(MAKE) -C $(dir) &&) true

.PHONY: all run
