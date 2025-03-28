# LLMart Command Line Reference

## Introduction

LLMart uses [Hydra](https://hydra.cc/) for configuration management, which provides a flexible command line interface for specifying configuration options. This document details all available command line arguments organized by functional groups.

Configurations can be specified directly on the command line (using dot notation for nested configurations) and are applicable to launching the *LLMart* module using `-m llmart`. Lists are specified using brackets and comma separation (careful with extra spaces). For example:

```bash
accelerate launch -m llmart model=llama3-8b-instruct data=basic steps=567 optim.n_tokens=11 banned_strings=[car,machine]
```

You can also compose configurations from pre-defined groups and override specific values as needed.
## Core Configuration

These parameters control the basic behavior of experiments. Parameters marked as *MISSING* are mandatory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | *MISSING* | Model name<br> Can be either one of the pre-defined options, or the Hugging Face name supplied to `AutoModelForCausalLM` |
| `revision` | string | *MISSING* | Model revision<br> Hugging Face revision supplied to `AutoModelForCausalLM`<br>⚠️ Mandatory only when the model is not one of the pre-defined ones |
| `data` | string | "advbench_behavior" | Dataset configuration<br> Can be one of the pre-defined options or `custom` if loading an arbitrary Hugging Face dataset|
| `loss` | string | "model" | Loss function type |
| `optim` | string | "gcg" | Optimization algorithm<br> Choices: `gcg, sgd, adam`<br>⚠️ Using `sgd` or `adam` will result in a soft embedding attack |
| `scheduler` | string | "linear" | Scheduler to use on an integer hyper-parameter specified by `scheduler.var_name`<br> Choices: `constant, linear, exponential, cosine, multistep, plateau` |
| `experiment_name` | string | `llmart` | Name of the folder where results will be stored |
| `output_dir` | string | `${now:%Y-%m-%d}/${now:%H-%M-%S.%f}` | Name of the sub-folder where results for the current run will be stored<br> Defaults to a millisecond-level timestamp |
| `seed` | integer | 2024 | Global random seed for reproducibility<br>⚠️ The seed will only reproduce results on the same number of GPUs as the original run |
| `use_deterministic_algorithms` | boolean | false | Whether to use cuDNN deterministic algorithms |
| `steps` | integer | 500 | Number of adversarial optimization steps |
| `early_stop` | boolean | true | Whether to enable early stopping<br> If `true`, enables early stopping once all forced tokens are rank-1 (guaranteed selection in greedy decoding) |
| `val_every` | integer | 50 | Validation frequency (in steps) |
| `max_new_tokens` | int | 512 | The maximum number of tokens to auto-regressively generate when periodically validating the adversarial attack |
| `save_every` | integer | 50 | Result saving frequency (in steps) |
| `per_device_bs` | integer | 1 | Per-device batch size<br> Setting this to `-1` will enable `auto` functionality for finding the largest batch size that can fit on the device<br>❗The value `-1` is currently only supported for single-device execution <br>⚠️ This parameter can greatly improve efficiency, but will error out if insufficient VRAM is available |
| `use_kv_cache` | boolean | false | Whether to use KV cache for efficiency<br>❗ Setting this to `true` is only intended for `len(data.subset)=1`, otherwise it may cause silent errors  |

## Model Configuration

Parameters related to model selection and configuration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model.task` | string | "text-generation" | Task for the model pipeline |
| `model.device` | string | "cuda" | Device to run on ("cuda", "cpu", etc.) |
| `model.device_map` | string | null | Device mapping strategy |
| `model.torch_dtype` | string | "bfloat16" | Torch data type |

**Pre-defined model options:**
- `llama3-8b-instruct`
- `llama3.1-8b-instruct`
- `llama3.1-70b-instruct`
- `llama3.2-1b-instruct`
- `llama3.2-11b-vision`
- `llamaguard3-1b`
- `llama3-8b-grayswan-rr`
- `deepseek-r1-distill-llama-8b`

## Attack & Optimization Configuration

Parameters for configuring adversarial token placement and optimization methods.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `banned_strings` | list[string] | empty | Any tokens that are superstrings of any element will be excluded from optimization<br> ⚠️ This could be useful for banning profanities from being optimized, although it is not sufficient to guarantee that the model cannot learn two adjacent tokens that decode to a banned string |
| `attack.suffix` | integer | 20 | How many adversarial suffix tokens are optimized |
| `attack.prefix` | integer | 0 | How many adversarial prefix tokens are optimized |
| `attack.pattern` | string | null | The string that is replaced by `attack.repl` tokens<br> Each occurence of the string pattern will be replaced with the same tokens |
| `attack.dim` | integer | 0 | The dimension out of `{0: dict_size, 1: embedding_dim}` used to define and compute gradients<br>⚠️ `0` is currently the only robust and recommended setting |
| `attack.default_token` | string | " !" | The initial string representation of the adversarial tokens<br>⚠️ If string here does not encode to a single token, the number of optimized tokens will be the length of the default token multiplied by `suffix` |
| `optim.lr` | float | 0.001 | Learning rate (step size) for the optimizer |
| `optim.n_tokens` | integer | 20 | Number of tokens to simultaneously optimize in a single step |
| `optim.n_swaps` | integer | 1024 | Number of token candidate swaps (replacements) to sample in a single step<br> |
| `scheduler.var_name` | string | "n_tokens" | The `optim` integer hyper-parameter that the scheduler modifies during optimization<br> Choices: `n_tokens, n_swaps` |

## Data Configuration

Parameters for data loading and processing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data.path` | string | *MISSING* | Name of Hugging Face dataset<br>Only required when using a `data=custom` |
| `data.subset` | list[integer] | null | Specific data samples to use from the dataset and learn a single adversarial attack for all of them |
| `data.files` | string | null | Files passed to the Hugging Face dataset |
| `data.shuffle` | boolean | false | Whether to shuffle data at each step |
| `data.n_train` | integer | 0 | Number of training samples to take from the `data.subset`<br> Leaving this and `data.{n_val, n_train}` to their default values will automatically use only the first sample for training and testing. |
| `data.n_val` | integer | 0 | Number of validation samples |
| `data.n_test` | integer | 0 | Number of test samples |
| `bs` | integer | 1 | Data batch size to use in an optimization step<br>⚠️ This is different than the core `per_device_bs` and must be equal to it if `len(data.subset) > 1`. |

**Pre-defined data options:**
- `basic`
- `advbench_behavior`
- `advbench_judge`
- `custom`
