import einops
import torch

import src.model.model
from src.process.task2_sliding_window import get_dataloader, sliding_3D_window

data_loader = get_dataloader()

model = src.model.model.UnetModel(1, 16, 6)

device = torch.device('cuda')
model.to(device)
for x, y in data_loader:
    y = torch.LongTensor(y.long())
    x, y = x.to(device), y.to(device)
    y = torch.nn.functional.one_hot(y, 16)
    y = einops.rearrange(y, 'b d w h c -> b c d w h')
    fun = sliding_3D_window
    for x_win, y_win in zip(fun(x, window_size=(32, 128, 128), step=(16, 64, 64)), fun(y, window_size=(32, 128, 128), step=(16, 64, 64))):
        print(x_win.shape)
        print(y_win.shape)
