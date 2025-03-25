"""
    -*- coding: utf-8 -*-
    Time    : 2025/3/25 21:01
    Author  : LazyShark
    File    : __init__.py.py
"""
"""
    dota标签转化yolo标签和裁剪均可以使用ultralytics的官方代码进行
    其中, 要先转化为yolo标签, 再进行裁剪
    转化格式如下, 要保证图片后缀为png, 如图片后缀不是png, 记得要修改脚本中的if png代码
    - DOTA
    ├─ images
    │   ├─ train
    │   └─ val
    └─ labels
        ├─ train_original
        └─ val_original
    转换:
    from ultralytics.data.converter import convert_dota_to_yolo_obb
    convert_dota_to_yolo_obb(r"C:/Users/LazyShark/Desktop/DOTA_original/dota")
    裁剪:
    from ultralytics.data.split_dota import split_test, split_trainval

    # split train and val set, with labels.
    split_trainval(
        data_root=r"C:/Users/LazyShark/Desktop/DOTA_original/dota",
        save_dir=r"C:/Users/LazyShark/Desktop/DOTA_original/dota-splot",
        rates=[0.5, 1.0, 1.5],  # multiscale
        gap=500,
    )
    # split test set, without labels.
    split_test(
        data_root=r"C:/Users/LazyShark/Desktop/DOTAv1.5/DOTA_original",
        save_dir=r"C:/Users/LazyShark/Desktop/DOTAv1.5/DOTAv1.5-split",
        rates=[0.5, 1.0, 1.5],  # multiscale
        gap=500,
    )
"""