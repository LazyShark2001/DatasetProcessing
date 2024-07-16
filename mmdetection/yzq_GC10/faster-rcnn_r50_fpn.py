_base_ = '../configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=5
        )
    )
)

# 修改数据集相关配置
data_root = r'C:/Users/LazyShark/Desktop/mmdetection/data/coco/'
metainfo = {
    'classes': ('youwu', 'jiaoban', 'huahen', 'fenbi', 'other'),
    # 'palette': [
    #     (220, 20, 60),
    # ]
}
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val/')))
test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_test2017.json',
        data_prefix=dict(img='test/')))

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = dict(ann_file=data_root + 'annotations/instances_test2017.json')

# optim_wrapper = dict(type='AmpOptimWrapper')
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
default_hooks = dict(logger=dict(type='LoggerHook', interval=50))

load_from='yzq/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'

# nohup python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_ciou_1x_visdrone.py > faster-rcnn-visdrone.log 2>&1 & tail -f faster-rcnn-visdrone.log
# python tools/test.py configs/faster_rcnn/faster-rcnn_r50_fpn_ciou_1x_visdrone.py work_dirs/faster-rcnn_r50_fpn_ciou_1x_visdrone/epoch_12.pth --show --show-dir test_save
# python tools/test.py configs/faster_rcnn/faster-rcnn_r50_fpn_ciou_1x_visdrone.py work_dirs/faster-rcnn_r50_fpn_ciou_1x_visdrone/epoch_12.pth --tta 