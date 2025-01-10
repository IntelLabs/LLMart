#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

from datetime import datetime
from omegaconf import DictConfig
from accelerate.utils import broadcast_object_list
from hydra.experimental.callback import Callback
from omegaconf import OmegaConf


class HydraFixedNow(Callback):
    """Hydra callback that ensures consistent "now" resolver.

    This is useful when using multi-process launchers, like accelerate, so that
    each process resolves the same datetime.
    """

    def __init__(self):
        self.datetime = None

    def on_run_start(self, config: DictConfig, **kwargs) -> None:
        self.datetime = broadcast_object_list([datetime.now()])[0]
        OmegaConf.register_new_resolver(
            "now",
            lambda pattern: self.datetime.strftime(pattern),  # type: ignore
            use_cache=True,
            replace=True,
        )

    def on_run_end(self, config: DictConfig, **kwargs) -> None:
        self.datetime = None
        OmegaConf.register_new_resolver(
            "now",
            lambda pattern: datetime.now().strftime(pattern),
            use_cache=True,
            replace=True,
        )
