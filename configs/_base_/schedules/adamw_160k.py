# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    )
)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10000,
    warmup_ratio=0.1,
    step=[100000, 140000])

runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(interval=16000)
evaluation = dict(interval=16000, metric=['mAP'])
