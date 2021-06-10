# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Created by:         Huy-Hien Vu (@hien-v)
Date created:       
Date last modified:
"""

import numpy as np
import torch

from fairseq import utils
from . import data_utils, FairseqDataset

def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    pass

