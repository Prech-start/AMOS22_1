import einops
import torch
from torch.nn.functional import pad
import src.model.model
from src.process.task2_sliding_window import get_dataloader, sliding_3D_window
from einops import rearrange
from src.train.loss import BCELoss_with_weight


def sliding_3D_window2(image, window_size, step):
    # image.shape = b, c, d, w, h
    # step must smaller than window_size
    image = rearrange(image, ' b c d w h -> b c w h d')
    _, _, width, height, depth = image.shape
    x_step, y_step, z_step = step
    x_window_size, y_window_size, z_window_size = window_size
    is_continue = False
    for z in range(0, depth, z_step):
        for y in range(0, height - y_window_size + 1, y_step):
            for x in range(0, width - x_window_size + 1, x_step):
                # 1 if z+z_window_size > depth
                # 2 if z.next() < depth -> z + step_z < depth
                # 3 step < window
                if z + z_window_size > depth and z - z_step + z_window_size != depth:
                    window = image[..., x:x_window_size + x, y:y_window_size + y, -1 - z_window_size:-1]
                else:
                    window = image[..., x:(x_window_size + x) if width > x_window_size else -1,
                             y:(y_window_size + y) if height > y_window_size else -1, z:z_window_size + z]
                # window = pad(window,)
                # _, _, w, h, d = window.shape
                # pad_w =
                # if window.shape[-1] == 16:
                #     print(window.shape)
                yield window


from tqdm import tqdm

data_loader = get_dataloader()

model = src.model.model.UnetModel(1, 16, 6)

device = torch.device('cuda')
model.to(device)
loss_weight = [1, 2, 2, 3, 6, 6, 1, 4, 3, 4, 7, 8, 10, 5, 4, 5]
criterion = BCELoss_with_weight(loss_weight)
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_error_window = 0
total_window = 0
for x, y in tqdm(data_loader):
    y = torch.LongTensor(y.long())
    y = torch.nn.functional.one_hot(y, 16)
    y = einops.rearrange(y, 'b d w h c -> b c d w h')
    fun = sliding_3D_window2
    for x_win, y_win in zip(fun(x, window_size=(128, 128, 64), step=(96, 96, 48)),
                            fun(y, window_size=(128, 128, 64), step=(96, 96, 48))):
        # x_batch = x[..., x_win[2]:x_win[3], x_win[4]:x_win[5], x_win[0]:x_win[1]]
        # y_batch = y[..., y_win[2]:y_win[3], y_win[4]:y_win[5], y_win[0]:y_win[1]]
        optimizer.zero_grad()
        x_batch, y_batch = x_win, y_win
        if x_batch.shape != (1, 1, 128, 128, 64):
            num_error_window += 1
            continue
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # print(x_batch.shape)
        # print(y_batch.shape)
        pred = model(x_batch)
        loss = criterion(pred, y_batch.float())
        loss.backward()
        # pred.cpu()
        # x_batch.cpu()
        # y_batch.cpu()
