#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

gpu ?= 0

all: run

run: run_basic run_autogcg run_fact_checking run_llmguard run_random_strings run_unlearning

run_autogcg:
	make -C autogcg gpu=$(gpu)
	-make -C autogcg clean

run_basic:
	make -C basic gpu=$(gpu)
	-make -C basic clean

run_fact_checking:
	make -C fact_checking gpu=$(gpu)
	-make -C fact_checking clean

run_llmguard:
	make -C llmguard gpu=$(gpu)
	-make -C llmguard clean

run_random_strings:
	make -C random_strings gpu=$(gpu)
	-make -C random_strings clean

run_unlearning:
	make -C unlearning gpu=$(gpu)
	-make -C unlearning clean

clean:
	-rm -rf $(patsubst %/,%,$(dir $(CURDIR)))/venv  $(patsubst %/,%,$(dir $(CURDIR)))/__pycache__

.PHONY: all run run_autogcg run_basic run_fact_checking run_llmguard run_random_strings run_unlearning clean
