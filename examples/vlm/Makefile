#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

include ../../preamble.mk

model=ByteDance-Seed/UI-TARS-2B-SFT
args=--model $(model) --max_epochs=3

all: run

run:
> $(RUN_GPU) --with-requirements requirements.txt main.py $(args)

.PHONY: all run
