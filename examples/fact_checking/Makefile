#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

include ../../preamble.mk

num_steps=2
args=--num_steps $(num_steps)

all: run

run:
> $(RUN_GPU) --with-requirements requirements.txt claim.py $(args)
> $(RUN_GPU) --with-requirements requirements.txt document.py $(args)

.PHONY: all run
