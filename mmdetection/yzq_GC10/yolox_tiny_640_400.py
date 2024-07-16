_base_ = '../configs/yolox/yolox_tiny_8xb8-300e_coco.py'

# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    bbox_head=dict(
        num_classes=10
    )
)

# 修改数据集相关配置
# dataset settings
data_root = 'C:/Users/LazyShark/Desktop/mmdetection/data/GC10_coco/'
dataset_type = 'CocoDataset'
metainfo = {
    'classes': ('punching_hole', 'welding_line', 'crescent_gap', 'water_spot', 'oil_spot', 'silk_spot', 'inclusion', 'rolled_pit', 'crease', 'waist folding'),
    # 'palette': [
    #     (220, 20, 60),
    # ]
}

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

img_scale = (640, 640)  # width, height


train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/'),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=backend_args),)

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric='bbox',
    backend_args=backend_args)

default_hooks = dict(logger=dict(type='LoggerHook', interval=50))
max_epochs = 400
num_last_epochs = 15
interval = 10

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)


# nohup python tools/train.py configs/yolox/yolox_tiny_8xb8-300e_visdrone.py > yolox-tiny-visdrone.log 2>&1 & tail -f yolox-tiny-visdrone.log
# python tools/test.py configs/yolox/yolox_tiny_8xb8-300e_visdrone.py work_dirs/yolox_tiny_8xb8-300e_visdrone/epoch_300.pth --show --show-dir test_save
# python tools/test.py configs/yolox/yolox_tiny_8xb8-300e_visdrone.py work_dirs/yolox_tiny_8xb8-300e_visdrone/epoch_300.pth --tta
# python tools/analysis_tools/get_flops.py configs/yolox/yolox_tiny_8xb8-300e_visdrone.py