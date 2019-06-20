import argparse

import numpy as np
from torch import nn

from erfnet_cp import customized_erfnet, erfnet
from gpu_energy_eval import GPUEnergyEvaluator
import time
import torch
import random


class CustomizedAlexnet(nn.Module):
    def __init__(self, width=None):
        super(CustomizedAlexnet, self).__init__()
        if width is None:
            width = [3, 64, 192, 384, 256, 256, 4096, 4096, 1000]
        self.features = nn.Sequential(
            nn.Conv2d(width[0], width[1], kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(width[1], width[2], kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(width[2], width[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width[3], width[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width[4], width[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(width[5] * 6 * 6, width[6]),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(width[6], width[7]),
            nn.ReLU(inplace=True),
            nn.Linear(width[7], width[8]),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def conv(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class CustomizedMobileNet(nn.Module):
    def __init__(self, width):
        super(CustomizedMobileNet, self).__init__()
        if width is None:
            width = [3, 32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024, 1001]
        assert len(width) == 16
        strides = [2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
        assert len(strides) == len(width) - 2

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        # print('Creating customized mobilenet...')
        conv_layers = []
        for i in range(len(strides)):
            conv_type = conv_bn if i == 0 else conv_dw
            conv_layers.append(conv_type(width[i], width[i+1], strides[i]))
            # print('{}--{}-->{}'.format(width[i], strides[i], width[i+1]))
        self.model = nn.Sequential(*conv_layers, nn.AvgPool2d(7))
        self.fc = nn.Linear(width[-2], width[-1])

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Energy Cost Data')
    parser.add_argument('--net', default='alexnet', help='network architecture')
    parser.add_argument('--num', type=int, default=10000, help='number of samples to generate')
    parser.add_argument('--gpuid', type=int, default=0, help='gpuid')
    parser.add_argument('--num_classes', type=int, default=1000, help='number of classes')
    parser.add_argument('--test_num', type=int, default=1000, help='number of repeated trails')
    parser.add_argument('--conv', action='store_true', help='only use conv layers')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--outfile', default='./output.npy', help='the output file of generated data')
    args = parser.parse_args()
    print(args.__dict__)

    if args.net == 'alexnet':
        net_class = CustomizedAlexnet
        width_ub = [64, 192, 384, 256, 256, 4096, 4096]
        h, w = 224, 224
    elif args.net == 'mobilenet':
        net_class = CustomizedMobileNet
        width_ub = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        h, w = 224, 224
    elif args.net == 'erfnet':
        net_class = customized_erfnet
        width_ub = erfnet().get_cpwub()
        h, w = 512, 1024
    else:
        raise NotImplementedError('not supported network architecture')
    test_num = args.test_num
    nclasses = args.num_classes
    fake_img = torch.randn([1, 3, h, w], dtype=torch.float32)
    random.seed(1)
    use_cuda = not args.cpu
    if use_cuda:
        fake_img = fake_img.cuda(device=args.gpuid)

    # test the upper-bound and lower-bound
    model = net_class(width=[3] + [1] * len(width_ub) + [nclasses])
    model.eval()
    if use_cuda:
        model.cuda(device=args.gpuid)
    with torch.no_grad():
        evaluator = GPUEnergyEvaluator(gpuid=args.gpuid)
        start_time = time.time()
        evaluator.start()
        for _ in range(test_num):
            output = model.forward(fake_img)
            torch.cuda.synchronize()
        energy_used = evaluator.end()
        time_used = time.time() - start_time
    print('Empty model: energy used {:.4e} J, time used {:.4e} s'.format(energy_used / float(test_num), time_used))

    model = net_class(width=[3] + width_ub + [nclasses])
    model.eval()
    if use_cuda:
        model.cuda(device=args.gpuid)
    with torch.no_grad():
        evaluator = GPUEnergyEvaluator(gpuid=args.gpuid)
        start_time = time.time()
        evaluator.start()
        for _ in range(test_num):
            output = model.forward(fake_img)
            torch.cuda.synchronize()
        energy_used = evaluator.end()
        time_used = time.time() - start_time
    print('Full model: energy used {:.4e} J, time used {:.4e} s'.format(energy_used / float(test_num), time_used))
    # collecting energy, time info data
    result = []
    # save the data as [width, energy, time]
    item = np.zeros(2 + len(width_ub) + 2, dtype='float32')

    width = [0] * len(width_ub)
    intervals = [max(round(w), 1) for w in width_ub]
    data_num = args.num
    wlb = [interval + 1 for interval in intervals]
    wub = [w+interval-1 for w, interval in zip(wlb, intervals)]
    print(wlb)
    print(wub)
    print('===========================')
    for i in range(data_num):
        width = [random.randint(w1, w2) for w1, w2 in zip(wlb, wub)]
        width = [3] + width + [nclasses]
        model = net_class(width=width)
        model.eval()
        if use_cuda:
            model.cuda(device=args.gpuid)

        with torch.no_grad():
            if args.conv:
                forward = model.conv
            else:
                forward = model.forward

            evaluator = GPUEnergyEvaluator(gpuid=args.gpuid)
            start_time = time.time()
            evaluator.start()
            for _ in range(test_num):
                output = forward(fake_img)
                torch.cuda.synchronize()
            energy_used = evaluator.end()
            time_used = time.time() - start_time
            item[:-2] = width
            item[-2] = energy_used / float(test_num)
            item[-1] = time_used / float(test_num)
            result.append(item.copy())

        if i % 100 == 99:
            print(item)
            print('saved {} items to {}'.format(i+1, args.outfile))
            np.save(args.outfile, np.stack(result))
    np.save(args.outfile, np.stack(result))
