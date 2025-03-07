#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import hydra
from omegaconf import OmegaConf, DictConfig
from accelerate import Accelerator, find_executable_batch_size

from llmart import config
from llmart import run_attack


@hydra.main(version_base=None, config_name="llmart")
def main(dict_config: DictConfig):
    cfg: config.LLMartConf = OmegaConf.to_object(dict_config)  # type: ignore
    
    if cfg.per_device_bs == "auto":
        @find_executable_batch_size(starting_batch_size=128)
        def loop_optimal_bs(batch_size):
            cfg.per_device_bs = batch_size
            cfg.validate_parameters()
            run_attack(cfg)
        return loop_optimal_bs()

    elif cfg.per_device_bs >= 1:
        cfg.validate_parameters()
        run_attack(cfg)

    else:
        print("Please enter the per_device_bs parameter correctly; enter an integer or set it to auto")
    

if __name__ == "__main__":
    main()
