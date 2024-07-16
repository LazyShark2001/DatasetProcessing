_base_ = '../configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'

checkpoint=None

model = dict(
    backbone=dict(
        deepen_factor=0.167,
        widen_factor=0.375,
        init_cfg=None),
    neck=dict(in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(num_classes=10)
)

# 修改数据集相关配置
data_root = 'C:/Users/LazyShark/Desktop/mmdetection/data/APDDD_coco/'
metainfo = {
    'classes': ('loudi', 'jupi', 'cahua', 'aoxian', 'budaodian', 'tufen', 'qikeng', 'tucengkailie', 'pengshang', 'zangdian'),
    # 'palette': [
    #     (220, 20, 60),
    # ]
}
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val/')))
test_dataloader = dict(
    batch_size=16,
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

default_hooks = dict(logger=dict(type='LoggerHook', interval=50))

max_epochs = 400


train_cfg = dict(max_epochs=max_epochs)



# nohup python tools/train.py configs/rtmdet/rtmdet_tiny_8xb32-300e_visdrone.py > rtmdet-tiny-visdrone.log 2>&1 & tail -f rtmdet-tiny-visdrone.log
# python tools/test.py configs/rtmdet/rtmdet_tiny_8xb32-300e_visdrone.py work_dirs/rtmdet_tiny_8xb32-300e_visdrone/epoch_300.pth --show --show-dir test_save
# python tools/test.py configs/rtmdet/rtmdet_tiny_8xb32-300e_visdrone.py work_dirs/rtmdet_tiny_8xb32-300e_visdrone/epoch_300.pth --tta
# python tools/analysis_tools/get_flops.py configs/rtmdet/rtmdet_tiny_8xb32-300e_visdrone.py