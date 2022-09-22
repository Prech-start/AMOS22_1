import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    输入的数据为 B * C * W * H * L

    编码器Encoder下采样 将数据卷积成 B * C * 1

    全连接层

    解码器Decoder上采样还原图像 恢复到 B * C * W * H * L

'''


class Model(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.button_channel = 128
        channel = [16, 32, 64, 128]
        self.encoder = Encoder(in_channel, channels=channel)
        self.decoder = Decoder(channels=channel, out_channel=out_channel)
        self.button = Button_model()
        # self.up = nn.Upsample(size=[224, 224, 16], mode='trilinear', align_corners=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        shape = x.shape
        # x = self.up(x)
        x, skip_connection = self.encoder(x)
        x = self.button(x)
        x = self.decoder(x, skip_connection, shape)
        # x = torch.round(x)
        x = self.softmax(x)
        return x


class Conv3d_block(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.Cov = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=out_channel,
                      kernel_size=(3, 3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channel, out_channels=out_channel,
                      kernel_size=(3, 3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.Cov(x)


# size=[1,224,224,16]
class Encoder(nn.Module):
    def __init__(self, in_channel, channels) -> None:
        super().__init__()
        self.Conv3D = nn.ModuleList()
        for channel in channels:
            self.Conv3D.append(Conv3d_block(in_channel=in_channel, out_channel=channel))
            in_channel = channel
        self.avg_pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

    def forward(self, x):
        skip_conection = []
        for down in self.Conv3D:
            x = down(x)
            # resnet
            x = self.avg_pool3d(x)
            skip_conection.append(x)
        return x, skip_conection


class Button_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


class CBAM(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


class Trans_FCN_block(nn.Module):
    def __init__(self, inchannel) -> None:
        super().__init__()
        self.con3d = Conv3d_block(in_channel=inchannel * 2, out_channel=inchannel // 2)
        # self.Channel_down = nn.Conv3d(in_channels = inchannel * 2, out_channels = inchannel//2 ,kernel_size=(1,1,1))
        # nn.ConvTranspose3d(in_channels=inchannel, out_channels=inchannel)

    def forward(self, x, skip_connection):
        shapes = skip_connection.shape

        x = nn.Upsample(size=shapes[2:], mode='trilinear', align_corners=True)(x)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.con3d(x)
        x = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels, out_channel) -> None:
        super().__init__()
        self.up_sample = nn.ModuleList()
        self.channels = channels
        for channel in channels[::-1]:
            self.up_sample.append(Trans_FCN_block(channel))
        self.down_channel = nn.Conv3d(channels[0] // 2, out_channel, kernel_size=(1, 1, 1))

    def forward(self, x, skip_connections, shape):
        for skip_connection, up_sample in zip(skip_connections[::-1], self.up_sample):
            x = up_sample(x, skip_connection)
        x = nn.Upsample(size=shape[2:], mode='trilinear', align_corners=True)(x)
        x = self.down_channel(x)
        # x = self.softmax(x)
        return x


# _*_ coding: utf-8 _*_
# Author: Jielong
# @Time: 21/08/2019 15:52
import sys
import time
import torch
import torch.nn as nn


# from unet3d_model.building_components import EncoderBlock, DecoderBlock
# sys.path.append("..")
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                                stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

    def forward(self, x):
        x = self.batch_norm(self.conv3d(x))
        # x = self.conv3d(x)
        x = F.elu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, model_depth=4, pool_size=2):
        super(EncoderBlock, self).__init__()
        self.root_feat_maps = 4
        self.num_conv_blocks = 2
        # self.module_list = nn.ModuleList()
        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.root_feat_maps
            for i in range(self.num_conv_blocks):
                # print("depth {}, conv {}".format(depth, i))
                if depth == 0:
                    # print(in_channels, feat_map_channels)
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
                else:
                    # print(in_channels, feat_map_channels)
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
            if depth == model_depth - 1:
                break
            else:
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling

    def forward(self, x):
        down_sampling_features = []
        for k, op in self.module_dict.items():
            if k.startswith("conv"):
                x = op(x)
                # print(k, x.shape)
                if k.endswith("1"):
                    down_sampling_features.append(x)
            elif k.startswith("max_pooling"):
                x = op(x)
                # print(k, x.shape)

        return x, down_sampling_features


class DecoderBlock(nn.Module):
    def __init__(self, out_channels, model_depth=4):
        super(DecoderBlock, self).__init__()
        self.num_conv_blocks = 2
        self.num_feat_maps = 4
        # user nn.ModuleDict() to store ops
        self.module_dict = nn.ModuleDict()

        for depth in range(model_depth - 2, -1, -1):
            # print(depth)
            feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps
            # print(feat_map_channels * 4)
            self.deconv = ConvTranspose(in_channels=feat_map_channels * 4, out_channels=feat_map_channels * 4)
            self.module_dict["deconv_{}".format(depth)] = self.deconv
            for i in range(self.num_conv_blocks):
                if i == 0:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 6, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
                else:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
            if depth == 0:
                self.final_conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=out_channels)
                self.module_dict["final_conv"] = self.final_conv

    def forward(self, x, down_sampling_features):
        """
        :param x: inputs
        :param down_sampling_features: feature maps from encoder path
        :return: output
        """
        for k, op in self.module_dict.items():
            if k.startswith("deconv"):
                x = op(x)
                x = torch.cat((down_sampling_features[int(k[-1])], x), dim=1)
            elif k.startswith("conv"):
                x = op(x)
            else:
                x = op(x)
        return x


class UnetModel2(nn.Module):

    def __init__(self, in_channels, out_channels, model_depth=4, final_activation="softmax"):
        super(UnetModel, self).__init__()
        self.encoder = EncoderBlock(in_channels=in_channels, model_depth=model_depth)
        self.decoder = DecoderBlock(out_channels=out_channels, model_depth=model_depth)
        self.decoder_centre = DecoderBlock(out_channels=1, model_depth=model_depth)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        # if final_activation == "sigmoid":
        #     self.sigmoid = nn.Sigmoid()
        # else:
        #     self.sigmoid = nn.Softmax(dim=1)

    def forward(self, x):
        x, downsampling_features = self.encoder(x)
        seg_x = self.decoder(x, downsampling_features)
        centre_x = self.decoder_centre(x, downsampling_features)
        seg_x = self.softmax(seg_x)
        centre_x = self.sigmoid(centre_x)
        # print("Final output shape: ", x.shape)
        return seg_x, centre_x

class UnetModel(nn.Module):

    def __init__(self, in_channels, out_channels, model_depth=4, final_activation="softmax"):
        super(UnetModel, self).__init__()
        self.encoder = EncoderBlock(in_channels=in_channels, model_depth=model_depth)
        self.decoder = DecoderBlock(out_channels=out_channels, model_depth=model_depth)
        # self.decoder_centre = DecoderBlock(out_channels=1, model_depth=model_depth)
        if final_activation == "sigmoid":
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = nn.Softmax(dim=1)

    def forward(self, x):
        x, downsampling_features = self.encoder(x)
        seg_x = self.decoder(x, downsampling_features)
        seg_x = self.sigmoid(seg_x)
        # print("Final output shape: ", x.shape)
        return seg_x

class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=2, padding=1, output_padding=1):
        super(ConvTranspose, self).__init__()
        self.conv3d_transpose = nn.ConvTranspose3d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=k_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   output_padding=output_padding)

    def forward(self, x):
        return self.conv3d_transpose(x)


if __name__ == '__main__':
    x = torch.rand(size=[1, 1, 64, 256, 256])
    # skip = torch.rand(size=[1, 256, 14, 14, 1])
    model = UnetModel2(1, 16, model_depth=4)
    # button shape 256,13,13,1
    y, z = model(x)
    print(y.shape, z.shape)
