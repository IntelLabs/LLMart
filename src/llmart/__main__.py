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
    '''
    def inner_loop():
        run_attack(cfg)
    inner_loop()
    '''
    
    #if(cfg.per_device_bs == "auto"):
    @find_executable_batch_size(starting_batch_size=32)
    def inner_training_loop(batch_size):
        cfg.per_device_bs = batch_size
        print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ  TESTING with batchsize " + str(cfg.per_device_bs) + "   ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
        run_attack(cfg)
    return inner_training_loop()
        
    #else:
    #    print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ" + str(cfg.per_device_bs) + "ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
    #    run_attack(cfg)
    

if __name__ == "__main__":
    main()
