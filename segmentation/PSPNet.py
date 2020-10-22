import torch
import torch.nn as nn
import torch.nn.functional as F

class PSPNet(nn.Module):
    def __init__(self, n_classes):
        super(PSPNet, self).__init__()

        block_config = [3, 4, 6, 3]  # resnet50
        img_size = 475
        img_size_8 = 60

        # コンストラクタ実装後引数入れる
        self.feature_conv = FeatureMap_convolution()
        self.feature_res_1 = ResidualBlockPSP()
        self.feature_res_2 = ResidualBlockPSP()
        self.feature_dilated_res_1 = ResidualBlockPSP()
        self.feature_dilated_res_2 = ResidualBlockPSP()

        self.pyramid_pooling = PylamidPooling()

        self.decode_feature = DecodePSPFeature()

        self.aux = AuxiliaryPSPlayers()

    def forward(self, x):
        x = self.feature_conv(x)
        x = self.feature_res_1(x)
        x = self.feature_res_2(x)
        x = self.feature_dilated_res_1(x)

        output_aux = self.aux(x)

        x = self.pyramid_pooling(x)
        output = self.decode_feature(x)

        return output, output_aux


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)

        return outputs


class FeatureMap_convolution(nn.Module):
    def __init__(self):
        super(FeatureMap_convolution, self).__init__()

        # Convolution1
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 3, 64, 3, 2, 1, 1, False
        self.cbnr_1 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # Convolution2
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 64, 3, 1, 1, 1, False
        self.cbnr_2 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # Convolution3
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 128, 3, 1, 1, 1, False
        self.cbnr_3 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # Max-pooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr_1(x)
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        output = self.maxpool(x)
        return output


class ResidualBlockPSP(nn.Sequential):
    pass


class Conv2DBatchNorm(nn.Module):
    pass


class bottleNeckPSP(nn.Module):
    pass


class bottleNeckIdentifyPSP(nn.Module):
    pass


class PylamidPooling(nn.Module):
    pass


class DecodePSPFeature(nn.Module):
    pass


class AuxiliaryPSPlayers(nn.Module):
    pass