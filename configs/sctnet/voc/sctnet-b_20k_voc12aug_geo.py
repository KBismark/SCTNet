
checkpoint_backbone = 'pretrain/SCT-B_Pretrain.pth'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder_Distill',
    pretrained=None,
    backbone=dict(
        type='SCTNet',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint_backbone
        ),
        base_channels=64,
        spp_channels=128),
    decode_head=dict(
        type='SCTHead',
        in_channels=256,
        channels=256,
        dropout_ratio=0.0,
        in_index=0,
        num_classes=21,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='AU_SCTHead',
            in_channels=128,
            channels=128,
            dropout_ratio=0.0,
            in_index=1,
            num_classes=21,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.4)),
        # dict(
        #     type='VitGuidanceHead',
        #     init_cfg=dict(
        #         type='Pretrained',
        #         checkpoint=checkpoint_teacher),
        #     in_channels=256,
        #     channels=256,
        #     base_channels=64,
        #     in_index=2,
        #     num_classes=21,
        #     loss_decode=dict(type='AlignmentLoss', loss_weight=[3, 15, 15, 15])),
        # Geometric Supervision 
        # in_index=0 taps the SAME x_out feature (256ch, stride 8) that
        # decode_head (SCTHead, in_index=0) reads.
        dict(
            type='GeoAuxHead',
            in_channels=256,
            channels=128,
            dropout_ratio=0.0,
            in_index=0,
            num_classes=21,
            align_corners=False,
            prior_loss_weight=0.15,
            boundary_loss_weight=0.10,
            boundary_dilate_size=2,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0)
        ),  # GeoAuxHead won't call self.losses()
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


dataset_type = 'PascalVOCDataset'
data_root = 'data/VOCdevkit/VOC2012'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir=['SegmentationClass', 'SegmentationClassAug'],
        split=[
            'ImageSets/Segmentation/train.txt',
            'ImageSets/Segmentation/aug.txt'
        ],
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline
    )
)

log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=0.0004,
    betas=(0.9, 0.999),
    weight_decay=0.0125,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            teacher_backbone=dict(lr_mult=0.0),
            teacher_head=dict(lr_mult=0.0)
        )
    )
)
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=1e-06,
    by_epoch=False
)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)
find_unused_parameters = True
auto_resume = True
seed = 1209821543
