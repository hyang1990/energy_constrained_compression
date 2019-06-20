import torch.nn as nn
import torch.nn.functional as F


class TFSamePad(nn.Module):
    def __init__(self, kernel_size, stride):
        super(TFSamePad, self).__init__()
        self.stride = stride
        if kernel_size != 3:
            raise NotImplementedError('only support kernel_size == 3')

    def forward(self, x):
        if self.stride == 2:
            return F.pad(x, (0, 1, 0, 1))
        elif self.stride == 1:
            return F.pad(x, (1, 1, 1, 1))
        else:
            raise NotImplementedError('only support stride == 1 or 2')


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, dropout=False):
        super(MobileNet, self).__init__()
        conv_bn = self.conv_bn
        conv_dw = self.conv_dw
        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
            nn.Dropout(0.2 if dropout else 0.0),
        )
        # self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

    @staticmethod
    def conv_bn(inp, oup, stride):
        return nn.Sequential(
            TFSamePad(3, stride),
            nn.Conv2d(inp, oup, 3, stride, 0, bias=False),
            nn.BatchNorm2d(oup, eps=0.001, momentum=0.001),
            nn.ReLU6(inplace=True)
        )

    @staticmethod
    def conv_dw(inp, oup, stride):
        return nn.Sequential(
            TFSamePad(3, stride),
            nn.Conv2d(inp, inp, 3, stride, 0, groups=inp, bias=False),
            nn.BatchNorm2d(inp, eps=0.001, momentum=0.001),
            nn.ReLU6(inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, eps=0.001, momentum=0.001),
            nn.ReLU6(inplace=True)
        )

    def depthwise_weights_name(self):
        res = set()
        for name, p in self.named_parameters():
            if name.endswith('weight') and p.dim() == 4 and p.shape[1] == 1:
                res.add(name)
        return res

    def l2wd_loss(self, weight_decay):
        res = 0.0
        weights_name = self.depthwise_weights_name()
        for name, p in self.named_parameters():
            if name.endswith('weight') and name not in weights_name:
                res += (p ** 2).sum()
        res *= 0.5 * weight_decay
        return res

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

    def get_inhw(self, x):
        res = []
        for module in self.model._modules.values():
            if isinstance(module, nn.Dropout):
                break
            res.append((x.size(2), x.size(3)))
            x = module(x)
        # x = x.view(-1, 1024)
        assert res[-1] == (7, 7)
        res[-1] = (1, 1)
        # x = self.fc(x)
        return res