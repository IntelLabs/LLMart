#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

include ../../preamble.mk

num_steps=2
args=--num_steps $(num_steps)
nltk=nltk==3.10.3

all: run

nltk_data:
> uv run --isolated --no-project --with $(nltk) -m nltk.downloader punkt_tab

claim: nltk_data
> $(RUN_GPU) --with-requirements requirements.txt --with $(nltk) claim.py $(args)

document: nltk_data
> $(RUN_GPU) --with-requirements requirements.txt --with $(nltk) document.py $(args)

run: claim document

.PHONY: all nltk_data claim document run
