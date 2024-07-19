训练参考https://blog.csdn.net/chao_xy/article/details/130179886
# 编译
python setup.py develop
pip install -v -e .
# 本质上只是掉包，-v表示安装过程输出详细信息， -e表示在可编辑模式下安装
# 训练准备
    coco数据集样式
    data
      -coco
        -annotations
          -instances_test2017.json
          -instances_train2017.json
          -instances_val2017.json
        -test2017
        -train2017
        -val2017
# 修改配置文件
1.修改mmdet/datasets/coco.py中
  
    class CocoDataset(BaseDetDataset)
          METAINFO = {
        'classes':
            ('crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches'),        # 填自己的类别
        'palette':
            []         #颜色
    }
2.修改mmdet/evaluation/functional/class_names.py中

    def coco_classes() -> list:
    """Class names of COCO."""
    return [
        'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches'      # 同上
    ]
至此，数据集类别问题解决，接下来调整实验参数

    1.在configs/_base_/models/faster-rcnn_r50_fpn.py中调整模型大小和num_classes
    
    2.在configs/_base_/datasets/coco_detection.py调整图像尺寸、batch_size、num_worker和验证部分是要测试集还是验证集
    
    3.在configs/_base_/schedules/schedule_1x.py中调整epochs数量和val_interval和优化器学习率等参数
    
    4.在configs/_base_/default_runtime.py调整一些工具参数，如画图、预训练权重等
# 训练样例
    参数
    --work-dir：指定训练保存模型和日志的路径
    --resume-from：从预训练模型chenkpoint中恢复训练
    --no-validate：训练期间不评估checkpoint
    --gpus：指定训练使用GPU的数量（仅适用非分布式训练）
    --gpu-ids： 指定使用哪一块GPU（仅适用非分布式训练）
    --seed：随机种子
    --deterministic：是否为CUDNN后端设置确定性选项
    --options： 参数字典
    --cfg-options: 如果指明，这里的键值对将会被合并到配置文件中。
    --launcher： {none,pytorch,slurm,mpi} job launcher
    --local_rank： LOCAL_RANK
    --autoscale-lr： 根据 GPU 数量自动缩放 LR
    --amp: 启用AMP精度混合训练
    compile：pytorch2.0以上可以使用compile训练加速，windows不支持
    --resume ${CHECKPOINT_FILE}: 从某个 checkpoint 文件继续训练.
    --cfg-options 'Key=value': 覆盖使用的配置文件中的其他设置.

 # 单卡
python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py --auto-scale-lr --amp --cfg-options compile=False

# 单机多卡
./tools/dist_train.sh configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py 2 --cfg-options compile=False

# 单机多卡+AMP
./tools/dist_train.sh configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py 2 --amp --auto-scale-lr --cfg-options compile=False

# 自定义
python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_build.py --auto-scale-lr --amp --cfg-options compile=False

./tools/dist_train.sh configs/faster_rcnn/faster-rcnn_r50_fpn_build.py 2 --amp --auto-scale-lr --cfg-options compile=False
————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

# faster-RCNN推理
python tools/test.py work_dirs/faster-rcnn_r50_fpn_1x_coco/faster-rcnn_r50_fpn_1x_coco.py work_dirs/faster-rcnn_r50_fpn_1x_coco/epoch_26.pth --show-dir test_result


# yolox
https://blog.csdn.net/kuyugoing/article/details/134888241

修改configs/yolox/yolox_s_8xb8-300e_coco.py中的内容,75、127、167、42行
75 classes = ('crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches')

183为epochs，191为学习率，252batchsize
# 启动
python tools/train.py configs/yolox/yolox_tiny_coco.py
# 推理
python tools/test.py work_dirs/yolox_tiny_coco/yolox_tiny_coco.py work_dirs/yolox_tiny_coco/epoch_300.pth --show-dir test_result


# CenterNet
修改configs/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco.py中的num_classes
# 启动
python tools/train.py configs/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco.py
# 推理
python tools/test.py configs/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco.py work_dirs/centernet-update_r50-caffe_fpn_ms-1x_coco/epoch_77.pth --show-dir test_result


 # 测试演示样例

    --show  决定是否现实图片
    --out  将结果输出为pkl格式的文件，该文件中会保存各个类别对应的信息，用于计算AP
    --show-dir  将测试得到的文件存到目标文件夹下
    --eval bbox  表示评估mAP,选择需要评估的指标，比如segm是分割的情况，这是mask rcnn网络会有这个结果，还有bbox等
    --options "classwise=True"  表示评估每个类别的AP
    --cfg-options: 如果指明，这里的键值对将会被合并到配置文件中
    RESULT_FILE: 结果文件名称，需以 .pkl 形式存储。如果没有声明，则不将结果存储到文件。
    mmdet/evaluation/metrics/coco_metric.py中修改classwise的值，True计算每个类别的mAP，iou在configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py配置文件中修改

 # 1 gpu
 python tools/test.py configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py pretrain/mask_rcnn_r50_fpn.pth --show
 # 保存结果
 python tools/test.py work_dirs/faster-rcnn_r50_fpn_1x_coco/faster-rcnn_r50_fpn_1x_coco.py work_dirs/faster-rcnn_r50_fpn_1x_coco/epoch_26.pth --show-dir test_result
 # 多 GPU
./tools/dist_test.sh configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py pretrain/mask_rcnn_r50_fpn.pth 2

# 自定义
./tools/dist_test.sh configs/mask_rcnn/mask-rcnn_UResnet50_fpn_build.py work_dirs/mask-rcnn_UResnet50_fpn_build/epoch_40.pth 2 --out ./result.pkl --show-dir test_build_result


——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
 # 图片演示样例
 python demo/image_demo.py demo/demo.jpg configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py --weights pretrain/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth

 ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
 # 实用工具

# 1 日志分析
    参数
    plot_curve：该参数后跟的是训练保存的json文件
    --keys：后面跟的是要绘制的损失关键字，可以跟多个值
    --out：后面跟的是绘制保存的结果，可以保存成png图片，也可以保存成pdf

# 绘制分类损失曲线图
python tools/analysis_tools/analyze_logs.py plot_curve ./work_dirs/mask-rcnn_r50_fpn_2x_build/20230831_163425/vis_data/20230831_163425.json --keys loss_cls --legend loss_cls
# 绘制分类损失、回归损失曲线图，保存图片为对应的 pdf 文件
python tools/analysis_tools/analyze_logs.py plot_curve ./work_dirs/faster-rcnn_r50_fpn_1x_coco/20231127_172043/vis_data/20231127_172043.json --keys loss coco/bbox_mAP_50 --out mAP.pdf
# 在相同图像中比较两次运行结果的 bbox mAP
python tools/analysis_tools/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
# 计算平均训练速度
python tools/analysis_tools/analyze_logs.py cal_train_time log.json [--include-outliers]

# 2 结果分析
    参数
    config: model config 文件的路径。
    prediction_path: 使用 tools/test.py 输出的 pickle 格式结果文件。
    show_dir: 绘制真实标注框与预测框的图像存放目录。
    --show：决定是否展示绘制 box 后的图片，默认值为 False。
    --wait-time: show 时间的间隔，若为 0 表示持续显示。
    --topk: 根据最高或最低 topk 概率排序保存的图片数量，若不指定，默认设置为 20。
    --show-score-thr: 能够展示的概率阈值，默认为 0。
    --cfg-options: 如果指定，可根据指定键值对覆盖更新配置文件的对应选项

# 测试 Mask R-CNN 并可视化结果，保存图片至 results/
python tools/analysis_tools/analyze_results.py configs/mask_rcnn/mask-rcnn_r50_fpn_2x_build.py result.pkl results --show
# 测试 Mask R-CNN 并指定 top-k 参数为 50，保存结果图片至 results/
python tools/analysis_tools/analyze_results.py configs/mask_rcnn/mask-rcnn_r50_fpn_2x_build.py result.pkl results --topk 50
# 如果你想过滤低概率的预测结果，指定 show-score-thr 参数
python tools/analysis_tools/analyze_results.py configs/mask_rcnn/mask-rcnn_r50_fpn_2x_build.py result.pkl results --show-score-thr 0.3

# 3 可视化
tensorboard --logdir=work_dirs/mask-rcnn_r50_gtlbfpn_build --port=8008
tensorboard --logdir=Experiment/cocobuild --port=8008


# 4 计算模型复杂度
python tools/analysis_tools/get_flops.py configs/mask_rcnn/mask-rcnn_r50_fpn_2x_build.py 

python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]

# 5 计算FPS (仅支持分布式训练)
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py work_dirs/faster-rcnn_r50_fpn_1x_coco/faster-rcnn_r50_fpn_1x_coco.py work_dirs/faster-rcnn_r50_fpn_1x_coco/epoch_27.pth --launcher pytorch

# 若推理过程中一直有图出现，更改配置文件中show的参数

# 精确率召回率，参数计算量FPS
https://blog.csdn.net/weixin_43722052/article/details/136598864
https://www.bilibili.com/video/BV17C41137dW/?spm_id_from=333.999.0.0
https://github.com/z1069614715/objectdetection_script/tree/master/mmdet-course

