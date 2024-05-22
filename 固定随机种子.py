import os
from random import random

import numpy as np
import torch

def moveimg(fileDir, tarDir):
    fileDir = fileDir + "\\"
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1

    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    for name in sample:
        shutil.copy(fileDir + name, tarDir + "\\" + name)  #复制
    return


def seed_torch(seed=42):
    # seed init
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置了Python的哈希种子，确保在使用散列的地方（例如集合或字典）也能产生确定性的结果
    # torch seed init.
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = True  # 启用了cuDNN的性能优化模式，它会根据硬件和输入数据动态地选择最快的CNN算法。但是，在追求确定性的情况下，通常会将其设为False，因为它会影响重现性
    torch.backends.cudnn.deterministic = True  # 启用了cuDNN的确定性模式，确保相同的输入能够得到相同的输出
    torch.backends.cudnn.enabled = True  # 启用了cuDNN，它是PyTorch用于GPU加速的核心库

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'  # 配置了cuBLAS库的工作空间大小，用于优化cuBLAS操作的性能，设置了每个GPU流的工作空间大小为16MB，并限制了每个流的工作空间分配次数为8次
    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)  # 调用启用了PyTorch的确定性算法，确保相同的输入会产生相同的输出，提高了代码的可重复性，可能影响训练速度
    print("seed:" + str(seed))