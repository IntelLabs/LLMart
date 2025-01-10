#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

from .text_generation import AdversarialTextGenerationPipeline
from .text_classification import AdversarialTextClassificationPipeline

__all__ = [
    "AdversarialTextGenerationPipeline",
    "AdversarialTextClassificationPipeline",
]
