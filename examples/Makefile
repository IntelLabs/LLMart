#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
# type: ignore
#
# This Makefile is used to run various example projects and clean up temporary files.

gpu ?= 0

all: run

run: run_basic run_autogcg run_fact_checking run_llmguard run_random_stinrgs run_unlearning

run_autogcg:
	make -C autogcg gpu=$(gpu)
	make -C autogcg clean

run_basic:
	make -C basic gpu=$(gpu)
	make -C basic clean

run_fact_checking:
	make -C fact_checking gpu=$(gpu)
	make -C fact_checking clean

run_llmguard:
	make -C llmguard gpu=$(gpu)
	make -C llmguard clean

run_random_stinrgs:
	make -C random_stinrgs gpu=$(gpu)
	make -C random_stinrgs clean

run_unlearning:
	make -C unlearning gpu=$(gpu)
	make -C unlearning clean

clean:
	rm -rf __pycache__ $(VENV_DIR)

.PHONY: all run run_autogcg run_basic run_fact_checking run_llmguard run_random_stinrgs run_unlearning clean