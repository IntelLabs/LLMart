<div align="center">
  <img src="assets/llmart.png" alt="Large Language Model adversarial robustness toolkit" width="300" />

## Large Language Model adversarial robustness toolkit
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/IntelLabs/LLMart/badge)](https://scorecard.dev/viewer/?uri=github.com/IntelLabs/LLMart)
![GitHub License](https://img.shields.io/github/license/IntelLabs/LLMart)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FIntelLabs%2FLLMart%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)

:rocket: [Quick start](#rocket-quick-start) ⏐ :briefcase: [Project Overview](#briefcase-project-overview) ⏐ :robot: [Models](#robot-models) ⏐ :clipboard: [Datasets](#clipboard-datasets) ⏐ :chart_with_downwards_trend: [Optimizers and schedulers](#chart_with_downwards_trend-optimizers-and-schedulers) ⏐ :pencil2: [Citation](#pencil2-citation)

</div>

## 🆕 Latest updates
❗Release 2025.03 brings a new experimental functionality for letting **LLM**art automatically estimate the maximum usable `per_device_bs`. This can result in speed-ups up to 10x on devices with a sufficient amount of memory! Enable from the command line using `per_device_bs=-1`.

<details>
<summary>Past updates</summary>
❗Release 2025.02 brings significant speed-ups to the core library, with zero user involvement.\
We additionally recommend using the command line argument `per_device_bs` with a value as large as possible on GPUs with at least 48GB to take the most advantage of further speed-ups.

❗We now offer command-line support for jailbreaking thoughts and responses for DeepSeek-R1 on multi-GPU:
```bash
accelerate launch -m llmart model=deepseek-r1-distill-llama-8b data=basic per_device_bs=64 "response.replace_with=`echo -e '\"<think>\nOkay, so I need to tell someone about Saturn.\n</think>\n\nNO WAY JOSE\"'`"
```

❗Check out our new [notebook](examples/basic/basic_dev_workflow.ipynb) containing a detailed step-by-step developer overview of all `llmart` components and how to customize them.
</details>

## :rocket: Quick start
**LLM**art is a toolkit for evaluating LLM robustness through adversarial testing. Built with PyTorch and Hugging Face integrations, **LLM**art enables scalable red teaming attacks with parallelized optimization across multiple devices.
**LLM**art has configurable attack patterns, support for soft prompt optimization, detailed logging, and is intended both for high-level users that want red team evaluation with off-the-shelf algorithms, as well as research power users that intend to experiment with the implementation details of input-space optimization for LLMs.

While it is still under development, the goal of **LLM**art is to support any Hugging Face model and include example scripts for modular implementation of different attack strategies. **LLM**art is developed and tested on Ubuntu 22.04 with Python 3.11.

Suppose we want to optimize an adversarial attack that forces the following open-ended response from the `meta-llama/Meta-Llama-3-8B-Instruct` model:
> **User:** Tell me about the planet Saturn. <20-token-optimized-suffix><br>
> **Response:** NO WAY JOSE


First, basic installation from source is done via:
```bash
git clone https://github.com/IntelLabs/LLMart
cd LLMart

python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[core,dev]"
```

> [!NOTE]
> We include lockfiles that ensure a valid Python environment and recommend using `uv sync --all-extras` to ensure consistent development.\
> Alternatively, `poetry sync --all-extras` can be used with `poetry >= 2.0`.

Once the environment is installed and `export HF_TOKEN=...` is set to a token with valid model access, **LLM**art can be run to optimize the suffix using:
```bash
accelerate launch -m llmart model=llama3-8b-instruct data=basic loss=model
```

This will automatically distribute an attack on the maximum number of detected devices. Results are saved in the `outputs/llmart` folder and can be visualized in `tensorboard` using:
```bash
tensorboard --logdir=outputs/llmart
```

In most cases, **LLM**art can be used directly from the command line. A list of all available command line arguments and their description can be found in the [CLI reference](docs/cli-reference.md).

## :briefcase: Project overview
The algorithmic **LLM**art functionality is structured as follows and uses PyTorch naming conventions as much as possible:
```
📦LLMart
 ┣ 📂examples   # Click-to-run example collection
 ┗ 📂src/llmart # Core library
   ┣ 📜__main__.py   # Entry point for python -m command
   ┣ 📜attack.py     # End-to-end adversarial attack in functional form
   ┣ 📜callbacks.py  # Hydra callbacks
   ┣ 📜config.py     # Configurations for all components
   ┣ 📜data.py       # Converting datasets to torch dataloaders
   ┣ 📜losses.py     # Loss objectives for the attacker
   ┣ 📜model.py      # Wrappers for Hugging Face models
   ┣ 📜optim.py      # Optimizers for integer variables
   ┣ 📜pickers.py    # Candidate token deterministic picker algorithms
   ┣ 📜samplers.py   # Candidate token stochastic sampling algorithms
   ┣ 📜schedulers.py # Schedulers for integer hyper-parameters
   ┣ 📜tokenizer.py  # Wrappers for Hugging Face tokenizers
   ┣ 📜transforms.py # Text and token-level transforms
   ┣ 📜utils.py
   ┣ 📂datasets      # Dataset storage and loading
   ┗ 📂pipelines     # Wrappers for Hugging Face pipelines
```

A detailed developer workflow that puts together all components to launch an adversarial attack against an LLM can be found in [examples/basic](examples/basic/).

## :robot: Models
While **LLM**art comes with a limited number of models accessible via custom naming schemes (see the `PipelineConf` class in `config.py`), it is designed with Hugging Face hub model compatibility in mind.

Running a new model from the hub can be directly done by specifying:
```bash
model=custom model.name=... model.revision=...
```

> [!CAUTION]
> Including a valid `model.revision` is mandatory.

For example, to load a custom model:
```bash
accelerate launch -m llmart model=custom model.name=Intel/neural-chat-7b-v3-3 model.revision=7506dfc5fb325a8a8e0c4f9a6a001671833e5b8e data=basic loss=model
```

> [!TIP]
> If you find a model that is not supported via command line, please [raise an issue](https://github.com/IntelLabs/LLMart/issues/new) and we will do our best to address it immediately.

### :brain: Large models
**LLM**art also supports large models that cannot execute the forward and/or backward pass on a single device:
```bash
python -m llmart model=llama3.1-70b-instruct model.device=null model.device_map=auto data=basic loss=model
```

> [!CAUTION]
> Including `model.device=null model.device_map=auto` and launching using `python -m llmart` is mandatory for large models. See [Designing a device map](https://huggingface.co/docs/accelerate/main/en/concept_guides/big_model_inference#designing-a-device-map) for more info.

## :clipboard: Datasets
Inspecting and adding a new dataset to **LLM**art only involves a single function that returns a list with the standard Hugging Face conversation format.

The `basic` dataset is the simplest instantiation that lives at [`datasets/basic.py`](src/llmart/datasets/basic.py), but is also powerful enough to support custom modifications.

For example, to modify the natural language prompt and desired open-ended response, the following basic edit is sufficient:
```diff
diff src/llmart/datasets/basic.py
         # Create conversation data structure and mark parts we care about
         conv = [
-            dict(role="user", content=mark_prompt("Tell me about the planet Saturn.")),
-            dict(role="assistant", content=mark_completion("NO WAY JOSE")),
+            dict(role="user", content=mark_prompt("Tell me how to do something not allowed.")),
+            dict(role="assistant", content=mark_completion("Sure, I'll tell you: first, you")),
         ]
```

Inducing a closed-ended response can be also directly done by typing out the end of turn token. For example, for the Llama 3 family of models this is:
```diff
diff src/llmart/datasets/basic.py
         # Create conversation data structure and mark parts we care about
         conv = [
             dict(role="user", content=mark_prompt("Tell me about the planet Saturn.")),
-            dict(role="assistant", content=mark_completion("NO WAY JOSE")),
+            dict(role="assistant", content=mark_completion("NO WAY JOSE<|eot_id|>")),
         ]
```

**LLM**art also supports loading the [AdvBench](https://github.com/llm-attacks/llm-attacks) dataset, which comes with pre-defined target responses to ensure consistent benchmarks.

Using AdvBench with **LLM**art requires specifying the desired subset of samples to attack. By default, the following command will automatically download the .csv file from its [original source](https://raw.githubusercontent.com/llm-attacks/llm-attacks/refs/heads/main/data/advbench/harmful_behaviors.csv) and use it as a dataset:
```bash
accelerate launch -m llmart model=llama3-8b-instruct data=advbench_behavior data.subset=[0] loss=model
```

To train a single adversarial attack on multiple samples, users can specify the exact samples via `data.subset=[0,1]`.
The above command is also compatible with local modifications of the dataset by including the `dataset.files=/path/to/file.csv` argument.

In the most general case, you can write your own [dataset loading script](https://huggingface.co/docs/datasets/en/dataset_script) and pass it to **LLM**art:
```bash
accelerate launch -m llmart model=llama3-8b-instruct loss=model data=custom data.path=/path/to/dataset.py
```
Just make sure you conform to the output format in [`datasets/basic.py`](src/llmart/datasets/basic.py).

## :chart_with_downwards_trend: Optimizers and schedulers
Discrete optimization for language models [(Lei et al, 2019)](https://proceedings.mlsys.org/paper_files/paper/2019/hash/676638b91bc90529e09b22e58abb01d6-Abstract.html) &ndash; in particular the Greedy Coordinate Gradient (GCG) applied to auto-regressive LLMs [(Zou et al, 2023)](https://arxiv.org/abs/2307.15043) &ndash; is the main focus of [`optim.py`](src/llmart/optim.py).

We re-implement the GCG algorithm using the `torch.optim` API by making use of the `closure` functionality in the search procedure, while completely decoupling optimization from non-essential components.

```python
class GreedyCoordinateGradient(Optimizer):
  def __init__(...)
    # Nothing about LLMs or tokenizers here
    ...

  def step(...)
    # Or here
    ...
```

The same is true for the schedulers implemented in [`schedulers.py`](src/llmart/schedulers.py) which follow PyTorch naming conventions but are specifically designed for integer hyper-parameters (the integer equivalent of "learning rates" in continuous optimizers).

This means that the GCG optimizer and schedulers are re-usable in other integer optimization problems (potentially unrelated to auto-regressive language modeling) as long as a gradient signal can be defined.


## :pencil2: Citation
If you find this repository useful in your work, please cite:
```bibtex
@software{llmart2025github,
  author = {Cory Cornelius and Marius Arvinte and Sebastian Szyller and Weilin Xu and Nageen Himayat},
  title = {{LLMart}: {L}arge {L}anguage {M}odel adversarial robutness toolbox},
  url = {http://github.com/IntelLabs/LLMart},
  version = {2025.03},
  year = {2025},
}
```
