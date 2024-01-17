# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional

import torch
from mmcv.runner import auto_fp16
from mmcv.utils import ConfigDict
from rsidet.models.builder import DETECTORS
from torch import Tensor

from .neg_rpn_query_support_detector import NegRPNQuerySupportDetector
from .meta_rcnn import MetaRCNN


@DETECTORS.register_module()
class NegRPNMetaRCNN(NegRPNQuerySupportDetector, MetaRCNN):
    pass
