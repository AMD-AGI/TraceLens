###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Vendored stub for moonshotai/Kimi-K3 `configuration_kimi_k3.py`.
# Present only so the fixture directory mirrors the real snapshot layout;
# TraceLens analyses `modeling_kimi_linear.py` alone (source-only) and never
# imports this module.
from transformers.configuration_utils import PretrainedConfig


class KimiLinearConfig(PretrainedConfig):
    model_type = "kimi_linear"
