import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Down_a(nn.Module):
    def __init__(self, in_channel):
        super(Down_a, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=in_channel, out_channels=in_channel * 2, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(in_channels=in_channel * 2, out_channels=in_channel * 4, kernel_size=3, stride=2,
                               padding=1)
        self.conv3 = nn.Conv3d(in_channels=in_channel * 4, out_channels=in_channel * 2, kernel_size=3, stride=2,
                               padding=1)
        self.conv4 = nn.Conv3d(in_channels=in_channel * 2, out_channels=in_channel, kernel_size=3, stride=2, padding=1)
        self.pooling = nn.AdaptiveAvgPool3d(output_size=2)
        self.linear1 = nn.Linear(8, 6)
        self.linear2 = nn.Linear(6, 3)

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.pooling(x))
        x = x.flatten().reshape(b, c, -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


if __name__ == '__main__':
    input_x = torch.randn((1, 16, 320, 160, 160))
    m = Down_a(16)
    print(input_x.shape)
    print(m(input_x).shape)
