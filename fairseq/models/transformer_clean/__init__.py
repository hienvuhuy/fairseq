# Copyright (c) Facebook Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .transformer_clean_config import (
    TransformerCleanConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from .transformer_clean_decoder import TransformerCleanDecoder, TransformerCleanDecoderBase, Linear
from .transformer_clean_encoder import TransformerCleanEncoder, TransformerCleanEncoderBase
# from .transformer_clean_models import (
#     TransformerCleanModel,
#     base_architecture,
#     # tiny_architecture,
#     # transformer_iwslt_de_en,
#     # transformer_wmt_en_de,
#     # transformer_vaswani_wmt_en_de_big,
#     # transformer_vaswani_wmt_en_fr_big,
#     # transformer_wmt_en_de_big,
#     # transformer_wmt_en_de_big_t2t,
# )
from .transformer_clean_models import *
from .transformer_clean_base import TransformerCleanModelBase, Embedding


__all__ = [
    "TransformerCleanModelBase",
    "TransformerCleanConfig",
    "TransformerCleanDecoder",
    "TransformerCleanDecoderBase",
    "TransformerCleanEncoder",
    "TransformerCleanEncoderBase",
    "TransformerCleanModel",
    "Embedding",
    "Linear",
    "DEFAULT_MAX_SOURCE_POSITIONS",
    "DEFAULT_MAX_TARGET_POSITIONS",
    "DEFAULT_MIN_PARAMS_TO_WRAP",
]
