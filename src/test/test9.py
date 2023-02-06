import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

from src.run_centre2 import UnetModel, get_train_data

model = UnetModel(1, 16, 6)

images, labels, _ = next(iter(get_train_data()))
writer.add_graph(model, images)
writer.close()