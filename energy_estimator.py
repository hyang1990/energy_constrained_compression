import argparse

import numpy as np
import torch
import torch.nn as nn
from erfnet_cp import erfnet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
from torch.nn.parameter import Parameter
import random

Alexnet_kernel_size = [11., 5., 3., 3., 3., 6., 1., 1.]
Alexnet_width_ub = [3, 64, 192, 384, 256, 256, 4096, 4096, 1000]

Mobilenet_kernel_size = [3., 3. / 32. + 1., 3. / 64 + 1., 3. / 128. + 1., 3. / 128. + 1., 3. / 256. + 1., 3. / 256. + 1.,
                         3. / 512. + 1., 3. / 512. + 1., 3. / 512. + 1., 3. / 512.+ 1., 3. / 512. + 1., 3. / 512. + 1.,
                         3. / 1024. + 1., 1.]
Mobilenet_width_ub = [3, 32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024, 1001]


class EnergyEstimateWidthRescale(nn.Module):
    def __init__(self, scales):
        super(EnergyEstimateWidthRescale, self).__init__()
        self.scales = Parameter(torch.tensor(scales, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        assert x.dim() != 1
        x = x / self.scales
        return torch.cat([(x[:, 0].detach() * x[:, 1]).unsqueeze(1),
                          x[:, 1:-2] * x[:, 2:-1],
                          (x[:, -2] * x[:, -1].detach()).unsqueeze(1)], dim=1)


class EnergyEstimateNet(nn.Module):
    def __init__(self, n_nodes=None, preprocessor=None):
        super(EnergyEstimateNet, self).__init__()
        if n_nodes is None:
            n_nodes = [len(Alexnet_width_ub) - 1, 1]  # linear model for Alexnet

        self.islinear = (len(n_nodes) == 2)
        # self.preprocessor = EnergyEstimateWidthRescale([384.0] * 6 + [4096.0] * 3)

        if preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = lambda x: x

        layers = []
        for i, _ in enumerate(n_nodes):
            if i < len(n_nodes) - 1:
                layer = nn.Linear(n_nodes[i], n_nodes[i + 1], bias=True)
                if len(n_nodes) == 2:
                    layer.weight.data.zero_()
                    layer.bias.data.zero_()
                layers.append(layer)
                if i < len(n_nodes) - 2:
                    layers.append(nn.SELU())
        self.regressor = nn.Sequential(*layers)

    def forward(self, x):
        single_data = (x.dim() == 1)
        if single_data:
            x = x.unsqueeze(0)
        res = self.regressor(self.preprocessor(x))
        if single_data:
            res = res.squeeze(0)
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Energy Estimator Training')
    parser.add_argument('--infile', default='./energy_alexnet.npy', help='the input file of training data')
    parser.add_argument('--outfile', default='./energymodel.pt', help='the output file of trained model')
    parser.add_argument('--net', default='alexnet', help='network architecture')
    parser.add_argument('--preprocess', default='rescale', help='preprocessor method')
    parser.add_argument('--batch_size', type=int, default=-1, help='input batch size for training')
    parser.add_argument('--seed', type=int, default=117, help='random seed (default: 117)')
    parser.add_argument('--epochs', type=int, default=10000, help='number of epochs to train')
    parser.add_argument('--wd', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--errhist', default=None, help='the output of error history')
    parser.add_argument('--pinv', action='store_true', help='use pseudo inverse to solve (only for bilinear model)')


    args = parser.parse_args()
    print(args.__dict__)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # create data loader
    data = np.load(args.infile)
    np.random.shuffle(data)
    val_portion = 0.2
    val_num = round(data.shape[0] * val_portion)

    tr_data, val_data = data[val_num:, :], data[:val_num, :]

    preprocess = lambda x: x

    tr_features, tr_labels = torch.from_numpy(preprocess(tr_data[:, :-2])), torch.from_numpy(tr_data[:, -2]).unsqueeze(
        1)
    val_features, val_labels = torch.from_numpy(preprocess(val_data[:, :-2])), torch.from_numpy(
        val_data[:, -2]).unsqueeze(1)

    if args.batch_size < 0:
        args.batch_size = tr_features.size(0)

    tr_loader = DataLoader(torch.utils.data.TensorDataset(tr_features, tr_labels), batch_size=args.batch_size,
                           shuffle=True)
    val_loader = DataLoader(torch.utils.data.TensorDataset(val_features, val_labels), batch_size=args.batch_size,
                            shuffle=False)


    def validate_model(model, loader, verbose=False):
        with torch.no_grad():
            model.eval()
            mrae = 0.0
            for data, label in loader:
                data = data.cuda()
                label = label.cuda()
                output = model(data)
                if verbose:
                    print(torch.cat([label, output], dim=-1))
                # mae += F.l1_loss(label, output)
                mrae += torch.mean(torch.abs(label - output) / torch.abs(label)).item()

            mrae /= len(val_loader)

        return mrae

    if args.net == 'alexnet':
        if args.preprocess == 'rescale':
            model = EnergyEstimateNet(n_nodes=[len(Alexnet_width_ub) - 1, 1],
                                      preprocessor=EnergyEstimateWidthRescale(scales=(Alexnet_width_ub)))
        else:
            raise NotImplementedError
    elif args.net == 'mobilenet':
        if args.preprocess == 'rescale':
            model = EnergyEstimateNet(n_nodes=[len(Mobilenet_width_ub) - 1, 1],
                                      preprocessor=EnergyEstimateWidthRescale(scales=(Mobilenet_width_ub)))
        else:
            raise NotImplementedError
    elif args.net == 'erfnet':
        if args.preprocess == 'rescale':
            width_ub = [3] + erfnet().get_cpwub() + [20]
            model = EnergyEstimateNet(n_nodes=[len(width_ub) - 1, 1],
                                      preprocessor=EnergyEstimateWidthRescale(scales=(width_ub)))
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if not args.pinv:
        model.cuda()

    # args.wd /= tr_features.shape[0]
    optimizer = torch.optim.Adam(model.regressor.parameters(), lr=1e-3, weight_decay=args.wd)
    # optimizer = torch.optim.SGD(model.regressor.parameters(), lr=1e-5, momentum=0.9, weight_decay=args.wd)
    # optimizer = torch.optim.RMSprop(model.regressor.parameters(), lr=1e-2)
    best_model = copy.deepcopy(model)
    best_mrae = float('inf')
    err_hist = []
    err_res = []
    if args.pinv:
        assert model.islinear
        model.cpu()
        weight = None
        bias = None

        X = torch.cat([torch.ones((tr_features.shape[0], 1), dtype=tr_features.dtype),
                       model.preprocessor(tr_features).data], dim=1)
        XtX = X.t().mm(X)
        Y = tr_labels.data
        XtY = X.t().mm(Y)
        w = torch.gesv(XtY, XtX + 0.5 * args.wd * torch.eye(XtX.shape[0], dtype=XtX.dtype))[0].t()
        print('linear system (XtX + 0.5(wd)I)w=XtY solved, w={}'.format(w))
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.copy_(w[:, 1:].data)
                m.bias.data.copy_(w[:, 0].data)
                break
        model.cuda()
    else:
        for epoch in range(args.epochs):
            model.train()
            for data, label in tr_loader:
                data = data.cuda()
                label = label.cuda()
                optimizer.zero_grad()
                output = model(data)
                mse = F.mse_loss(label, output)
                # sys.stdout.write('{:.4e}===>  '.format(mse))
                mse.backward()
                optimizer.step()
                # sys.stdout.flush()

        val_mrae = validate_model(model, loader=val_loader)

        err_hist.append(val_mrae)
        tr_mrae = 0#validate_model(model, loader=tr_loader)
        if val_mrae < best_mrae:
            best_model.load_state_dict(model.state_dict())
            best_mrae = val_mrae
        print("epoch {}, lr={}: tr_mrae={:.4e}, val_mrae={:.4e} (Best:{:.4e})"
              .format(epoch, optimizer.param_groups[0]['lr'], tr_mrae, val_mrae, best_mrae))

    print('val_mre={:.4e}'.format(validate_model(model, loader=val_loader)))
    print(model.state_dict())
    with torch.no_grad():
        model.eval()
        res = 0.0
        for data, label in val_loader:
            data = data.cuda()
            label = label.cuda()
            output = model(data)
            for i in range(label.size(0)):
                err_res.append((output[i].item(), label[i].item()))

    torch.save(model.state_dict(), args.outfile)
    if args.errhist is not None:
        with open(args.errhist, 'w') as f:
            for item in err_hist:
                f.write("%s\n" % item)

        with open(args.errhist + '.points', 'w') as f:
            for item in err_res:
                f.write("%s, %s\n" % (item[0], item[1]))
