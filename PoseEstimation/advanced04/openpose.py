import torch
import torch.nn as nn
from torch.nn import init
import torchvision


class OpenPoseNet(nn.Module):
    def __init__(self):
        super(OpenPoseNet, self).__init__()

        # Feature module
        self.model0 = OpenPose_Feature()

        # Stage module
        # models for pafs
        self.models1 = []
        # models for heatmaps
        self.models2 = []

        for i in range(1, 7):
            self.models1.append(make_OpenPose_block(f'block{i}_1'))

        for i in range(1, 7):
            self.models2.append(make_OpenPose_block(f'block{i}_2'))

    def forward(self, x):
        saved_for_loss = []

        # Feature module
        out0 = self.model0(x)

        # prepare output variables
        out = out0.copy()
        out1 = out.copy()
        out2 = out.copy()

        # Stage module
        for model1, model2 in zip(self.models1, self.models2):
            out1 = model1(out)
            out2 = model2(out)
            saved_for_loss.append(out1.copy())
            saved_for_loss.append(out2.copy())
            out = torch.cat([out1, out2, out0], 1)

        return (out1, out2), saved_for_loss


class OpenPose_Feature(nn.Module):
    def __init__(self):
        super(OpenPose_Feature, self).__init__()

        vgg19 = torchvision.models.vgg19(pretrained=True)
        model = {}
        model['block0'] = vgg19.features[0:23]

        model['block0'].add_module("23", torch.nn.Conv2d(
            512, 256, kernel_size=3, stride=1, padding=1))
        model['block0'].add_module("24", torch.nn.ReLU(inplace=True))
        model['block0'].add_module("25", torch.nn.Conv2d(
            256, 128, kernel_size=3, stride=1, padding=1))
        model['block0'] = model['block0']

        self.model = model['block0']

    def forward(self, x):
        outputs = self.model(x)
        return outputs


def make_OpenPose_block(block_name):
    blocks = {}
    # Stage1
    blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L1': [512, 38, 1, 1, 0]}]

    blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L2': [512, 19, 1, 1, 0]}]

    for i in range(2, 7):
        blocks[f'block{i}_1'] = [
            {f'Mconv1_stage{i}_L1': [185, 128, 7, 1, 3]},
            {f'Mconv2_stage{i}_L1': [128, 128, 7, 1, 3]},
            {f'Mconv3_stage{i}_L1': [128, 128, 7, 1, 3]},
            {f'Mconv4_stage{i}_L1': [128, 128, 7, 1, 3]},
            {f'Mconv5_stage{i}_L1': [128, 128, 7, 1, 3]},
            {f'Mconv6_stage{i}_L1': [128, 128, 1, 1, 0]},
            {f'Mconv7_stage{i}_L1': [128, 38, 1, 1, 0]}
        ]

        blocks[f'block{i}_2'] = [
            {f'Mconv1_stage{i}_L2': [185, 128, 7, 1, 3]},
            {f'Mconv2_stage{i}_L2': [128, 128, 7, 1, 3]},
            {f'Mconv3_stage{i}_L2': [128, 128, 7, 1, 3]},
            {f'Mconv4_stage{i}_L2': [128, 128, 7, 1, 3]},
            {f'Mconv5_stage{i}_L2': [128, 128, 7, 1, 3]},
            {f'Mconv6_stage{i}_L2': [128, 128, 1, 1, 0]},
            {f'Mconv7_stage{i}_L2': [128, 19, 1, 1, 0]}
        ]

    cfg_dict = blocks[block_name]

    layers = []

    for i in range(len(cfg_dict)):
        for k, v in cfg_dict[i].items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3], padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]

    net = nn.Sequential(*layers[:-1])

    def _initialize_weights_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

    net.apply(_initialize_weights_norm)

    return net
