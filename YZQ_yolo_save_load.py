'''
先弄出一个只保存模型权重，不保存模型名字的有序列表pt
model.model.state_dict()与之对齐
将权重列表的值赋给上述列表
model['ema'].load_state_dict(model1['model'].state_dict())


from ultralytics import YOLO
import torch

def are_keys_identical(dict1, dict2):
    # 获取两个字典的键列表
    keys1 = list(dict1.keys())
    keys2 = list(dict2.keys())

    # 比较两个键列表是否相等
    return keys1 == keys2

if __name__ == '__main__':
    # Load a model
    model = YOLO('runs/detect/train37/weights/last.pt')  # 也可以加载你自己的模型
    model_ = YOLO('runs/detect/train26/weights/last.pt')
    model1 = torch.load('runs/detect/train26/weights/last.pt')
    a = model.model.state_dict()
    b = model_.model.state_dict()
    c = are_keys_identical(a,b)
    model.model.load_state_dict(model1['model'].state_dict())
    # Validate the model
    metrics = model.val(split='val', iou=0.7, batch=16, data='RZB/RZB.yaml')
    metrics = model.val(split='test', iou=0.7, batch=16, data='RZB/RZB.yaml')
    metrics.box.map    # 查看目标检测 map50-95 的性能
    metrics.box.map50  # 查看目标检测 map50 的性能
    metrics.box.map75  # 查看目标检测 map75 的性能
    metrics.box.maps   # 返回一个列表包含每一个类别的 map50-95

'''

'''
真相是随便改，将模型的类被数调好即可

在ultralytics/engine/trainer.py中

if RANK in {-1, 0}:
    LOGGER.info(self.progress_string())
    pbar = TQDM(enumerate(self.train_loader), total=nb)
self.tloss = None
self.save_model()  #加上
for i, batch in pbar:
    self.run_callbacks("on_train_batch_start")
    # Warmup
    ni = i + nb * epoch
第一轮没训练完时，把权重文件夹的其他result.csv复制进去



from ultralytics import YOLO

if __name__ == '__main__':

    # model = YOLO("yaml/ALLDyConv+ALLMEWblock.yaml") # 从头开始构建新模型
    model = YOLO('yaml/FDADNet.yaml').load("runs/detect/train26/weights/last.pt") # 从头开始构建新模型

    # 使用模型
    model.train(data="RZB/RZB.yaml", epochs=400, device=[0])  # 训练模型

'''
import torch

if __name__ == '__main__':

    model = torch.load('runs/detect/train37/weights/best.pt')
    model1 = torch.load('runs/detect/train26/weights/last.pt')

    model['ema'].load_state_dict(model1['model'].state_dict())
    model['ema'].float()
    torch.save(model,'yzq.pt')


