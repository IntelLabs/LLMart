import copy
import fire  # type: ignore[reportMissingImports]
import numpy as np
from datetime import datetime
from omegaconf import OmegaConf

from hydra import compose, initialize
from ray import tune, train  # type: ignore[reportMissingImports]
from ray.tune.search.hebo import HEBOSearch  # type: ignore[reportMissingImports]

from llmart.attack import run_attack


# Experiment as closure
def experiment(config: dict) -> None:
    # Non-override parameters
    nonoverrides = ["num_seeds", "subset"]

    # Convert dictionary to list of hydra overrides
    overrides = [
        f"{key}={value}" for key, value in config.items() if key not in nonoverrides
    ]

    # Metrics to report
    reports = {}
    with initialize(version_base=None):
        test_losses = []
        for seed in range(config["num_seeds"]):
            local_overrides = copy.deepcopy(overrides)
            local_overrides.extend(
                [
                    f"seed={seed}",
                    f"data.subset=[{config['subset']}]",
                ]
            )
            # Load defaults and overrides
            hydra_cfg = compose(config_name="llmart", overrides=local_overrides)

            # Generate timestamp-based values
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            hydra_cfg.output_dir = "/tmp"
            hydra_cfg.experiment_name = f"{timestamp}"
            # Convert to config
            cfg = OmegaConf.to_object(hydra_cfg)

            # Run experiment and store results
            outputs = run_attack(cfg)  # type: ignore
            test_losses.append(outputs["attack/loss"].cpu().numpy())
            reports.update({f"loss_seed{seed}": outputs["attack/loss"].cpu().numpy()})
            reports.update({f"eval/prompt_seed{seed}": outputs["eval/test_prompt_0"]})
            reports.update(
                {f"eval/continuation_seed{seed}": outputs["eval/test_continuation_0"]}
            )

        # Compute 10th percentile loss across seeds for the sample
        loss = np.percentile(test_losses, q=10)

    reports.update({"loss": loss})
    train.report(reports)


def main(
    subset: int, per_device_bs: int = 64, steps: int = 50, num_seeds: int = 10
) -> None:
    # Define search space
    search_space = {
        "model": "llama3-8b-instruct",
        "data": "advbench_behavior",
        "per_device_bs": per_device_bs,
        "subset": subset,
        "steps": steps,
        "num_seeds": num_seeds,
        "optim.n_tokens": tune.randint(lower=1, upper=21),
        "scheduler": "plateau",
        "scheduler.factor": tune.uniform(lower=0.25, upper=0.9),
        "scheduler.patience": tune.randint(lower=1, upper=20),
        "scheduler.threshold": tune.uniform(lower=0.0, upper=0.25),
    }

    # Algorithm
    hebo = HEBOSearch(metric="loss", mode="min")

    tuner = tune.Tuner(
        tune.with_resources(experiment, resources={"gpu": 1}),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            time_budget_s=int(3600 * 2), num_samples=-1, search_alg=hebo
        ),
        run_config=train.RunConfig(name=f"autogcg_sample{subset}"),
    )
    results = tuner.fit()

    # Display best result
    print(results.get_best_result(metric="loss", mode="min"))


if __name__ == "__main__":
    fire.Fire(main)
