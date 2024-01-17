_base_ = [
    '../../../../../_base_/datasets/fine_tune_based/few_shot_nwpu.py',
    '../../../../../_base_/schedules/adamw_2k.py',
    '../../../../tfa_maskrcnn_r50.py',
    '../../../../../_base_/default_runtime.py'
]
seed = 0
shots = 20
num_classes = 10
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        type='FewShotNWPUDataset',
        # ann_cfg=[dict(method='TFA', setting='SPLIT1_3SHOT')],
        num_novel_shots=shots,
        num_base_shots=None,
        classes='ALL_CLASSES_SPLIT1',
        save_dataset=True,
        save_dataset_path=f'work_dirs/data_infos/nwpu-split1_{shots}shot_seed{seed}.json',
        balance_base_novel=True
    ),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1')
)

expr_name = f'st-fsod_maskrcnn_r50_nwpu-split1_seed{seed}_{shots}shot_fine-tuning'
init_kwargs = {
    'project': 'rsi-fewshot',
    'entity': 'tum-tanmlh',
    'name': expr_name,
    'resume': 'never'
}

wandb_cfg = dict(
    **init_kwargs,
    root = 'work_dirs',
    bbox_score_thr = [0.1] * num_classes,
    interval = 1,
    train_vis_thr = 0.0,
    max_vis_boxes = 30,
    num_eval_images = 300
)

model = dict(
    type='STTFA',
    backbone=dict(
        # depth=101,
        # frozen_stages=1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    frozen_parameters=[
        'backbone',
        # 'neck',
        # 'rpn_head.rpn_reg',
        # 'roi_head.bbox_head.shared_fcs',
        'rpn_head.rpn_conv.weight',
        'rpn_head.rpn_conv.bias',
        'rpn_head.rpn_cls.weight',
        'rpn_head.rpn_cls.bias',
        'rpn_head.rpn_reg.weight',
        'rpn_head.rpn_reg.bias',
        'rpn_head.rpn_conv_tch.weight',
        'rpn_head.rpn_conv_tch.bias',
        'rpn_head.rpn_cls_tch.weight',
        'rpn_head.rpn_cls_tch.bias',
        'rpn_head.rpn_reg_tch.weight',
        'rpn_head.rpn_reg_tch.bias',
        'roi_head.bbox_head', # freeze the teacher net
    ],

    rpn_head=dict(
        type='STRPNHeadV3',
        st_thre=1.0,
        bg_thre=0.8,
        drop_rate=0.3,
        alpha=0.999,
        neg_nms_type='normal',
        novel_class_inds=[7, 8, 9],
        init_cls=True,
        apply_loss_base=True,
        local_iter=2000
    ),
    roi_head=dict(
        type='STRoIHead',
        alpha=0.999,
        local_iter=2000,
        stu_bbox_head=dict(
            type='CosineSimSTBBoxHead',
            num_shared_fcs=2,
            num_classes=10,
            scale=20,
            drop_rate=0.3,
            st_thre=0.8,
            bg_thre=0.0,
            novel_class_inds=[7, 8, 9]
        ),
        bbox_head=dict(
            num_classes=10
        ),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.2,
                min_pos_iou=0.2,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=1.0,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.2,
                min_pos_iou=0.2,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler_pos=dict(
                type='RandomSampler',
                num=64,
                pos_fraction=1.0,
                neg_pos_ub=0,
                add_gt_as_proposals=True),
            sampler_neg=dict(
                type='RandomSampler',
                num=64,
                pos_fraction=0.0,
                neg_pos_ub=64,
                add_gt_as_proposals=False),
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            neg_filter_thre=0.4 # background boxes with scores higher than this threshold will be removed
        )
    ),
    wandb_cfg=wandb_cfg
)

# base model needs to be initialized with following script:
# tools/detection/misc/initialize_bbox_head.py
# please refer to README.md for more details.
load_from = ('work_dirs/st-fsod_maskrcnn_r50_10k_nwpu-split1_randomized_head/base_model_random_init_bbox_head.pth')

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(type='RSIDetWandbHook',
             init_kwargs=init_kwargs,
             interval=11,
             log_checkpoint=False,
             log_checkpoint_metadata=False,
             num_eval_images=300,
             bbox_score_thr=[0.9889138, 0.8887519, 0.9976034, 0.99616814, 0.99507606,
                             0.9735409803, 0.99172556, 0.8909433, 0.9743807, 0.950369],
             eval_after_run=False)
    ])

evaluation = dict(
    metric=['mAP'],
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'],
    # metric_items=['mAP_50'], iou_thrs=[0.5]
)
