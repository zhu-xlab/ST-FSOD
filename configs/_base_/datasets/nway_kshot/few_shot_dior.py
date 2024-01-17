# dataset settings
# img_norm_cfg = dict(
#     mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
pad_cfg = dict(img=(128, 128, 128), masks=0, seg=255)
pad_cfg_with_mask = dict(img=(128, 128, 128, 0), masks=0, seg=255)

image_size = (800, 800)
crop_size = (608, 608)

train_multi_pipelines = dict(
    query=[
        dict(type='LoadImageFromFile', to_float32=True),
        dict(type='LoadAnnotations', with_bbox=True),
        # dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='RandomFlip', flip_ratio=[0.5, 0.5], direction=['horizontal', 'vertical']),
        dict(type='RandomRotate90', prob=1.0),
        # large scale jittering
        dict(
            type='Resize',
            img_scale=image_size,
            ratio_range=(0.5, 2.0),
            multiscale_mode='range',
            keep_ratio=True),
        # dict(
        #     type='RandomCrop',
        #     crop_size=crop_size,
        #     crop_type='absolute',
        #     recompute_bbox=True,
        #     allow_negative_crop=False),
        dict(
            type='MinIoURandomCrop',
            min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            min_crop_size=0.3),
        # dict(type='Pad', size=crop_size, pad_val=pad_cfg),
        dict(type='Pad', size_divisor=32, pad_val=pad_cfg),
        dict(type='DefaultFormatBundle', img_to_float=True),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    ],
    support=[
        dict(type='LoadImageFromFile', to_float32=True),
        dict(type='LoadAnnotations', with_bbox=True),
        # dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='GenerateMask', target_size=(224, 224)),
        dict(type='RandomFlip', flip_ratio=[0.5, 0.5], direction=['horizontal', 'vertical']),
        dict(type='RandomRotate90', prob=1.0),
        # large scale jittering
        # dict(
        #     type='Resize',
        #     img_scale=image_size,
        #     ratio_range=(0.5, 2.0),
        #     multiscale_mode='range',
        #     keep_ratio=True),
        # dict(
        #     type='RandomCrop',
        #     crop_size=crop_size,
        #     crop_type='absolute',
        #     recompute_bbox=True,
        #     allow_negative_crop=True),
        dict(type='Pad', size_divisor=32, pad_val=pad_cfg_with_mask),
        dict(type='DefaultFormatBundle', img_to_float=True),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    ]
)
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
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# classes splits are predefined in FewShotVOCDataset
# data_root = 'data/VOCdevkit/'
datapipe_name = 'DIOR'
datapipe_root = '../../Datasets/Dataset4EO/DIOR'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type='NWayKShotDataset',
        num_support_ways=20,
        num_support_shots=1,
        one_support_shot_per_image=False,
        num_used_support_shots=30,
        save_dataset=True,
        dataset=dict(
            type='FewShotDIORDataset',
            ann_cfg=[
                dict(
                    type='datapipe',
                    datapipe_name=datapipe_name,
                    datapipe_root=datapipe_root,
                    datapipe_split='train'
                ),
            ],
            multi_pipelines=train_multi_pipelines,
            classes=None,
            instance_wise=False,
            dataset_name='query_support_dataset'),
    ),
    val=dict(
        type='FewShotDIORDataset',
        ann_cfg=[
            dict(
                type='datapipe',
                datapipe_name=datapipe_name,
                datapipe_root=datapipe_root,
                datapipe_split='val'
            ),
        ],
        # img_prefix=data_,
        pipeline=test_pipeline,
        classes=None),
    test=dict(
        type='FewShotDIORDataset',
        ann_cfg=[
            dict(
                type='datapipe',
                datapipe_name=datapipe_name,
                datapipe_root=datapipe_root,
                datapipe_split='val'
            ),
        ],
        # img_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        classes=None),
    model_init=dict(
        copy_from_train_dataset=True,
        samples_per_gpu=16,
        workers_per_gpu=8,
        type='FewShotDIORDataset',
        ann_cfg=None,
        # img_prefix=data_root,
        pipeline=train_multi_pipelines['support'],
        use_difficult=False,
        instance_wise=True,
        num_novel_shots=None,
        classes=None,
        dataset_name='model_init_dataset'))
# evaluation = dict(interval=5000, metric='mAP')
