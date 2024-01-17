# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseFewShotDataset
from .builder import build_dataloader, build_dataset
from .coco import COCO_SPLIT, FewShotCocoDataset
from .dataloader_wrappers import NWayKShotDataloader
from .dataset_wrappers import NWayKShotDataset, QueryAwareDataset
from .pipelines import CropResizeInstance, GenerateMask
from .utils import NumpyEncoder, get_copy_dataset_type
from .voc import VOC_SPLIT, FewShotVOCDataset
from .dior import DIOR_SPLIT, FewShotDIORDataset
from .dumpsite import DUMPSITE_SPLIT, FewShotDumpsiteDataset
from .nwpu import NWPU_SPLIT, FewShotNWPUDataset
from .dior_caption import FewShotDIORCaptionDataset
from .isaid import ISAID_SPLIT, FewShotISAIDDataset

__all__ = [
    'build_dataloader', 'build_dataset', 'QueryAwareDataset',
    'NWayKShotDataset', 'NWayKShotDataloader', 'BaseFewShotDataset',
    'FewShotVOCDataset', 'FewShotCocoDataset', 'CropResizeInstance',
    'GenerateMask', 'NumpyEncoder', 'COCO_SPLIT', 'VOC_SPLIT',
    'get_copy_dataset_type', 'DIOR_SPLIT', 'FewShotDIORDataset',
    'FewShotISAIDDataset', 'FewShotDIORCaptionDataset',
    'NWPU_SPLIT', 'FewShotNWPUDataset', 'FewShotDumpsiteDataset', 'DUMPSITE_SPLIT'
]
