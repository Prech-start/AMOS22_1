import numpy as np
from skimage.morphology import erosion, dilation
from skimage.morphology import square
import torch
import cv2

centre = np.zeros(shape=[21, 21])
centre[10, 10] = 10

print(centre)

selem = square(5)
selem2 = square(4)
dil = dilation(centre, selem)
dil = dilation(dil, selem2)
print(dil)
# kernel = np.zeros(shape=[21, 21])
# kernel[10, 10] = 1.0
# max_distance = 10 * pow(2, 1 / 2)
# e_ = 1e-4
# for (i, j), value in np.ndenumerate(kernel):
#     distance = np.linalg.norm(np.array([i, j]) - np.array([10, 10]))
#     kernel[i, j] = (max_distance - distance) / max_distance
#     print(distance)
#     pass
# print(kernel)
#
# dil = dilation(centre, kernel)
# print(dil)
class_xy = np.array(np.where(centre == 0))
distance_for_xyz = np.linalg.norm(class_xy.T - [10, 10], axis=1)  # N * 1
distance_list = distance_for_xyz[np.where(distance_for_xyz < 10)]
centre[tuple(class_xy.T[np.where(distance_for_xyz < 10)].T.tolist())] = 10 - distance_list
pass