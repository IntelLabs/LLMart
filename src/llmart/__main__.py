#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from accelerate import find_executable_batch_size

from llmart import config
from llmart import run_attack


@hydra.main(version_base=None, config_name="llmart")
def main(dict_config: DictConfig):
    cfg: config.LLMartConf = OmegaConf.to_object(dict_config)  # type: ignore

    if cfg.per_device_bs == -1:
        # Currently only supported for single-device execution
        if torch.cuda.device_count() > 1 or torch.xpu.device_count() > 1:
            raise ValueError(
                "Using 'auto' batch size is not yet supported for multi-device execution!"
            )

        @find_executable_batch_size(starting_batch_size=128)
        def loop_optimal_bs(batch_size: int):
            cfg.per_device_bs = batch_size
            run_attack(cfg)

        loop_optimal_bs()  # type: ignore[reportCallIssue]

    else:
        run_attack(cfg)


if __name__ == "__main__":
    main()
