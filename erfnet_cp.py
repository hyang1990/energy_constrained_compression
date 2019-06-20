# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated, cpw=None):
        super().__init__()
        if cpw is not None:
            assert len(cpw) == 3
        else:
            cpw = [chann] * 3

        self.cpwub = [chann] * 3
        self.chann = chann
        self.dilated = dilated
        self.hws = [None] * 3
        self.conv3x1_1 = nn.Conv2d(chann, cpw[0], (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(cpw[0], cpw[1], (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(cpw[1], eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(cpw[1], cpw[2], (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(cpw[2], chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

        def hw_hook(m, x):
            m.hws = [(x[0].size(2), x[0].size(3))] * 3

        self.register_forward_pre_hook(hw_hook)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)

    def get_cpw(self):
        return [self.conv1x3_1.in_channels, self.conv3x1_2.in_channels, self.conv1x3_2.in_channels]

    def set_cpw(self, cpwub):
        assert len(cpwub) == 3
        self.conv3x1_1 = nn.Conv2d(self.chann, cpwub[0], (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(cpwub[0], cpwub[1], (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(cpwub[1], eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(cpwub[1], cpwub[2], (3, 1), stride=1, padding=(1 * self.dilated, 0), bias=True,
                                   dilation=(self.dilated, 1))

        self.conv1x3_2 = nn.Conv2d(cpwub[2], self.chann, (1, 3), stride=1, padding=(0, 1 * self.dilated), bias=True,
                                   dilation=(1, self.dilated))


class Encoder(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__()
        dropoutprob = 0.3 if dropout else 0.0
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, dropoutprob, 2))
            self.layers.append(non_bottleneck_1d(128, dropoutprob, 4))
            self.layers.append(non_bottleneck_1d(128, dropoutprob, 8))
            self.layers.append(non_bottleneck_1d(128, dropoutprob, 16))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


# ERFNet
class ERFNet(nn.Module):
    def __init__(self, num_classes=20, encoder=None, dropout=False):  # use encoder to pass pretrained encoder
        super().__init__()
        self.dropout = dropout
        if encoder is None:
            self.encoder = Encoder(num_classes, dropout)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)  # predict=False by default
            return self.decoder.forward(output)

    def get_cpwub(self):
        res = []
        for m in self.modules():
            if isinstance(m, non_bottleneck_1d):
                res += m.cpwub
        return res

    def set_cpw(self, width):
        i = 0
        assert width[0] == 3
        i += 1
        for m in self.modules():
            if isinstance(m, non_bottleneck_1d):
                m.set_cpw(width[i:i+3])
                i += 3
        assert i == len(width) - 1
        self.decoder.output_conv = nn.ConvTranspose2d(16, width[i], 2, stride=2, padding=0, output_padding=0, bias=True)

    def get_cp_weights(self):
        res = [None]  # the first layer is not pruned
        for m in self.modules():
            if isinstance(m, non_bottleneck_1d):
                res += [m.conv1x3_1.weight, m.conv3x1_2.weight, m.conv1x3_2.weight]
        return res

    def get_inhw(self, input):
        self.forward(input)
        res = [None]
        for m in self.modules():
            if isinstance(m, non_bottleneck_1d):
                assert m.hws[0] is not None and m.hws[1] is not None and m.hws[2] is not None
                res += m.hws
        return res


def erfnet(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ERFNet(**kwargs)
    if pretrained:
        cityscapes_pretrained_erfnet_path = \
            os.path.dirname(os.path.realpath(__file__)) + '/pretrained/erfnet_pretrained_cityscapes.pth'

        def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
            own_state = model.state_dict()
            # print(len(own_state.keys()))
            # print(len(state_dict.keys()))
            for name, param in state_dict.items():
                # print(name)
                if name not in own_state:
                    if name.startswith("module."):
                        own_state[name.split("module.")[-1]].copy_(param)
                    else:
                        print(name, " not loaded")
                        continue
                else:
                    own_state[name].copy_(param)
            return model

        load_my_state_dict(model, torch.load(cityscapes_pretrained_erfnet_path))

    return model


def customized_erfnet(width):
    model = erfnet(pretrained=False)
    model.set_cpw(width)
    return model