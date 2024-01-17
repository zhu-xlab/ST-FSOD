# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union
import pdb

from rsidet.models.builder import DETECTORS
from rsidet.models.detectors.two_stage import TwoStageDetector


@DETECTORS.register_module()
class TFA(TwoStageDetector):
    """Implementation of `TFA <https://arxiv.org/abs/2003.06957>`_"""

    """
    def train_step(self, data: Dict, optimizer: Union[object, Dict]) -> Dict:
        pdb.set_trace()
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        # For most of query-support detectors, the batch size denote the
        # batch size of query data.
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data['query_data']['img_metas']))

        return outputs
    """
