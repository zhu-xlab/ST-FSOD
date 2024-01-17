_base_ = ['../_base_/models/mask_rcnn_r50_fpn.py']
model = dict(
    type='TFA',
    frozen_parameters=[
        'backbone', 'neck', 'rpn_head', 'roi_head.bbox_head.shared_fcs'
    ],
    roi_head=dict(
        mask_roi_extractor=None,
        mask_head=None,
        bbox_head=dict(
            type='CosineSimBBoxHead',
            num_shared_fcs=2,
            num_classes=20,
            scale=20)
    )
)
