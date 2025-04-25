#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import hydra
from omegaconf import OmegaConf, DictConfig

from llmart import config
from llmart import run_attack


@hydra.main(version_base=None, config_name="llmart")
def main(dict_config: DictConfig):
    cfg: config.LLMartConf = OmegaConf.to_object(dict_config)  # type: ignore

    run_attack(cfg)


if __name__ == "__main__":
    main()
