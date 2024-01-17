_base_ = [
    '../../../_base_/datasets/fine_tune_based/base_dior-trainval_bs16.py',
    '../../../_base_/schedules/adamw_40k.py',
    '../../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
data = dict(
    train=dict(classes='BASE_CLASSES_SPLIT4'),
    val=dict(classes='BASE_CLASSES_SPLIT4'),
    test=dict(classes='BASE_CLASSES_SPLIT4'))
# lr_config = dict(warmup_iters=100, step=[12000, 16000])
# runner = dict(max_iters=18000)
# model settings
model = dict(
    type='MaskRCNN',
    backbone=dict(
        depth=101,
        # frozen_stages=1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')
    ),
    roi_head=dict(
        mask_roi_extractor=None,
        mask_head=None,
        bbox_head=dict(num_classes=15)))

# using regular sampler can get a better base model
use_infinite_sampler = False
