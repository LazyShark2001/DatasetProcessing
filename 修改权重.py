"""
    -*- coding: utf-8 -*-
    Time    : 2025/5/2 12:50
    Author  : LazyShark
    File    : 修改权重.py.py
"""
"""
    torch网络在组合时, 只关心自身self的变量名, 不关注模块名字
"""
import torch
yzq = torch.load('opt.pth_path')

from collections import OrderedDict

# 创建新的有序字典
new_state_dict = OrderedDict()

# 按新顺序添加参数，跳过要删除的模块
for name, param in yzq.items():
    if 'fuse' not in name:
        new_state_dict[name] = param
    else:
        if '0.temp' not in name:
            name = name.replace("fuse.1", "fuse.0", 1)
            new_state_dict[name] = param

# 保存新模型
torch.save(new_state_dict, 'yzq.pt')