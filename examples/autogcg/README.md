## Basics and requirements
Install `llmart`and download/navigate to this folder. Run `uv pip install -r requirements.txt` in the working environment.


# `autoGCG` with `llmart`
The example in this folder shows how to integrate `LLMart` with the [ray-tune](https://docs.ray.io/en/latest/tune/index.html) hyperparameter optimization library to automatically search for the best attack hyper-parameters across one or multiple samples, given a total compute budget.

We call this functionality `autoGCG` -- automated greedy coordinate descent.

To run `autoGCG` on the `i`-th sample of the [AdvBench behavior](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv) dataset execute:
```bash
uv run python main.py --subset i
```

The script will automatically use the maximum number of GPUs and parallelize hyper-parameter tuning for the `n_tokens` hyper-parameter of GCG using `llmart`'s a [ChangeOnPlateauInteger](../../src/llmart/schedulers.py#L279) scheduler.
> [!NOTE]
> The default parameter `"per_device_bs": 64` may add too much memory pressure on GPUs with less than 48 GB of VRAM. If OOM errors occur, lowering `per_device_bs` should fix the issue.

Given the sensitivity of GCG with respect to seeding (random swap picking during optimization), `autoGCG` exploits this by minimizing the 10-percentile loss across ten different seeds, for the same sample.

By default, the optimization runs for a total of _two wall-clock hours_, regardless of how many GPUs are available:
```python
tune_config = tune.TuneConfig(
    time_budget_s=int(3600 * 2), num_samples=-1, search_alg=hebo
)
```


# Viewing results
The `ray.tune` experiment will be saved at the default location of `~/ray_results/autogcg_sample{i}`, after which it can be analyzed using [`tune.Tuner.restore`](https://docs.ray.io/en/latest/tune/examples/tune_analyze_results.html).
> [!NOTE]
> Properly using `tune.Tuner.restore` will require importing the experiment function as `from main import experiment` and passing it as an argument.
