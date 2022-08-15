from einops import rearrange


def sliding_3D_window(image, window_size, step):
    # image.shape = b, c, d, w, h
    # step must smaller than window_size
    image = rearrange(image, ' b c d w h -> b c w h d')
    _, _, width, height, depth = image.shape
    x_step, y_step, z_step = step
    x_window_size, y_window_size, z_window_size = window_size
    is_continue = False
    for z in range(0, depth, z_step):
        if is_continue:
            continue
        for y in range(0, height - y_window_size + 1, y_step):
            for x in range(0, width - x_window_size + 1, x_step):
                # 在第三个维度上，策略采用为不抛弃任何一个像素
                # 当 window 框超出目标图像范围且上一次的window取值不是紧贴最底部时，取最底部的图像
                # 1 if z+z_window_size > depth
                # 2 if z.next() < depth -> z + step_z < depth
                # 3 step < window
                if z + z_window_size > depth and z - z_step + z_window_size != depth:
                    window = image[..., x:x_window_size + x, y:y_window_size + y, -1 - z_window_size:-1]
                    # 若取最底层的window则跳过剩余对z的迭代
                    is_continue = True
                else:
                    window = image[..., x:x_window_size + x, y:y_window_size + y, z:z_window_size + z]
                yield window


import numpy as np

x = np.random.random(size=(1, 16, 128, 256, 256))
sliding_3D_window(x, (128, 128, 12), (6, 6, 6))
# print(x[..., :, :, -1 - 128: -1].shape)
