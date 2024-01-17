# dataset settings
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
pad_cfg = dict(img=(128, 128, 128), masks=0, seg=255)

image_size = (800, 800)
crop_size = (608, 608)

data_root = '../../Datasets/Dataset4EO/iSAID_patches/'

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=[0.5, 0.5], direction=['horizontal', 'vertical']),
    dict(type='RandomRotate90', prob=1.0),
    # large scale jittering
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.5, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=crop_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    # dict(type='Pad', size=crop_size, pad_val=pad_cfg),
    dict(type='Pad', size_divisor=32, pad_val=pad_cfg),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# classes splits are predefined in FewShotDIORDataset
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='FewShotISAIDDataset',
        save_dataset=False,
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file = '../../Datasets/Dataset4EO/iSAID_patches/train/instancesonly_filtered_train.json'
                # ann_file = '../../Datasets/Dataset4EO/iSAID_patches/val_1k/instancesonly_filtered_val_1k.json'
            ),
        ],
        img_prefix=data_root + 'train/images',
        # img_prefix=data_root + 'val_1k/images',
        pipeline=train_pipeline,
        classes=None,
        instance_wise=False
    ),
    val=dict(
        type='FewShotISAIDDataset',
        save_dataset=False,
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file = '../../Datasets/Dataset4EO/iSAID_patches/val/instancesonly_filtered_val.json'
                # ann_file = '../../Datasets/Dataset4EO/iSAID_patches/val_1k/instancesonly_filtered_val_1k.json'
            ),
        ],
        img_prefix=data_root + 'val/images',
        # img_prefix=data_root + 'val_1k/images',
        pipeline=test_pipeline,
        classes=None,
        instance_wise=True
    ),
    test=dict(
        type='FewShotISAIDDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file = '../../Datasets/Dataset4EO/iSAID_patches/val_1k/instancesonly_filtered_val_1k.json'
            ),
        ],
        img_prefix=data_root + 'val/images',
        pipeline=test_pipeline,
        test_mode=True,
        classes=None,
    )
)
