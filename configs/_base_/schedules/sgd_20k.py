# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    step=[15000, 18000])

runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(interval=2000)
evaluation = dict(interval=2000, metric=['mAP'])
