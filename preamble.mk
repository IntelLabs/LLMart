# Using guidelines from https://tech.davis-hansson.com/p/make/
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

ifeq ($(origin .RECIPEPREFIX), undefined)
	$(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

RUN_GPU=uv run --extra dev --extra gpu --isolated
RUN_XPU=uv run --extra dev --extra xpu --isolated
