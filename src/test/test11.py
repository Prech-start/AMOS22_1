import numpy as np
import torch

# ===================================================
m = torch.nn.Linear(20, 30)  # 输入特征数、输出特征30
weight_ = m.weight * m.weight
# m.weight = torch.rand(30,20)
# torch.Tensor = torch.FloatTensor 默认float类型
m.weight = torch.nn.Parameter(weight_)  # 自定义权值初始化
opti = torch.optim.Adam(m.parameters())
crit = torch.nn.CrossEntropyLoss()

for i in range(100):
    x = torch.ones(128, 20)  # N = 128组,特征20
    y = torch.ones(128, 30)  # N = 128组,特征20

    opti.zero_grad()
    out = m(x)

    loss = crit(out, y)
    loss.backward()

    opti.step()
    print(torch.min(m.weight))
    # print(list(m.parameters())) # 查看参数
