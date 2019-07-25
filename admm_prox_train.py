import argparse

import numpy as np
import os

import time
import torch
import random
import sys
import math

import misc
from erfnet_cp import erfnet
from misc import model_snapshot

from energy_estimator import Alexnet_width_ub, EnergyEstimateNet, Mobilenet_width_ub, EnergyEstimateWidthRescale
from torch import nn as nn
from torch.nn.parameter import Parameter

from utils import get_data_loaders, get_net_model, eval_loss_acc1_acc5, PlotData, joint_loss, \
    column_sparsity_common, array1d_repr, column_sparsity_mbnet, column_sparsity_resnet, eval_loss_iou, \
    joint_loss_cityscape
from torchvision import transforms


class EnergyConstrainedADMM_P(nn.Module):
    def __init__(self, net_model, width_ub):
        super(EnergyConstrainedADMM_P, self).__init__()
        self.net = net_model
        self.s = Parameter(torch.Tensor(width_ub))


class EnergyConstrainedADMM_D(nn.Module):
    def __init__(self, n_layers, z_init=0.0, y_init=0.0):
        super(EnergyConstrainedADMM_D, self).__init__()
        self.z = Parameter(torch.tensor(float(z_init)))
        self.y = Parameter(torch.Tensor(n_layers))
        self.y.data.fill_(y_init)

    def get_param_dicts(self, zlr, ylr):
        return [{'params': self.z, 'lr': zlr},
                {'params': self.y, 'lr': ylr}]


def admm_w_update_prox_common(p_model, d_model, optimizer, rho_y):
    i = 0
    last_output_width = None
    assert len(optimizer.param_groups) == 1
    alpha = optimizer.param_groups[0]['lr']
    if isinstance(optimizer, torch.optim.Adam):
        eps = optimizer.param_groups[0]['eps']
    elif isinstance(optimizer, torch.optim.SGD):
        eps = None
    else:
        raise NotImplementedError

    for name, p in p_model.net.named_parameters():
        if name.endswith('weight') and p.dim() > 1:
            if eps is not None:
                denom = optimizer.state[p]['exp_avg_sq'].sqrt().add_(eps)
            else:
                denom = None
            p = p.data
            s_i = float(p_model.s.data[i])
            y_i = float(d_model.y.data[i])
            input_width = p.size(1)
            if last_output_width is not None and input_width != last_output_width:
                # the first fc layer after conv layers
                assert input_width > last_output_width and input_width % last_output_width == 0
                input_width = last_output_width
            p_t = p.data.transpose(0, 1).contiguous()
            p_t_old_shape = p_t.shape
            p_t = p_t.view(input_width, -1)
            if denom is not None:
                denom = denom.data.transpose(0, 1).contiguous().view(input_width, -1)
                a = torch.sum((p_t ** 2) * denom, dim=1).data
            else:
                a = torch.sum(p_t ** 2, dim=1).data

            # print('i={}, y={}, p_shape={}'.format(i, y_i, p.shape))
            if rho_y == 0.0:
                a_rho = 0.0
            else:
                _, indices = torch.sort(a, descending=True)
                a_rank = torch.zeros_like(a)
                a_rank[indices] = torch.arange(1, a.numel() + 1, device=a_rank.device, dtype=a_rank.dtype)
                a_rho = rho_y * alpha * (torch.clamp(a_rank - s_i, min=0.0) ** 2
                                         - torch.clamp(a_rank - s_i - 1, min=0.0) ** 2)

            # project y
            if y_i > 0.0:
                a_topk, _ = torch.topk(a - a_rho, k=int(math.floor(s_i)), largest=True, sorted=True)
                y_i = min(y_i, a_topk[-1].item() / (2 * alpha))
                d_model.y.data[i] = y_i
            z_idx = (a < a_rho + 2 * alpha * y_i)

            # assert p_t.size(0) >= s_i, "error: {} < {}".format(p_t.size(0), s_i)
            if p.dim() == 4:
                p.data[:, z_idx, :, :] = 0.0
            else:
                if input_width == p.size(1):
                    p.data[:, z_idx] = 0.0
                else:
                    p_t[z_idx, :] = 0.0
                    p.data.copy_(p_t.view(*p_t_old_shape).transpose(0, 1))
            last_output_width = p.size(0)
            i += 1

    assert i + 1 == p_model.s.numel(), "eror:{} != {}".format(i + 1, p_model.s_scaled.numel())


def admm_w_update_prox_mbnet(p_model, d_model, optimizer, rho_y):
    # the first layer is normal conv, the last layer is fc, grouped conv are in the middle
    nlayers = len(Mobilenet_width_ub) - 1
    W = []
    assert len(optimizer.param_groups) == 1
    alpha = optimizer.param_groups[0]['lr']
    if isinstance(optimizer, torch.optim.Adam):
        eps = optimizer.param_groups[0]['eps']
    elif isinstance(optimizer, torch.optim.SGD):
        eps = None
    else:
        raise NotImplementedError

    p_scale = []
    for name, p in p_model.net.named_parameters():
        if name.endswith('weight') and p.dim() > 1:
            W.append(p)

    for m in p_model.net.modules():
        if isinstance(m, nn.BatchNorm2d):
            p_scale.append((m.weight.data / torch.sqrt(m.running_var.data + m.eps)).view(-1, 1, 1, 1))

    assert (len(W) - 2) / 2 + 2 == nlayers == p_model.s.numel() - 1, "error:{} != {}, {}".format((len(W) - 2) / 2 + 2,
                                                                                                 nlayers,
                                                                                                 p_model.s.numel())
    # debug_norms = []
    # debug_norms2 = []
    for layer_idx in range(nlayers):
        i = layer_idx * 2 - 1
        s_layer_idx = float(p_model.s.data[layer_idx])
        y_layer_idx = float(d_model.y.data[layer_idx])
        # print('layer={}, inner layer={}, s={}, y={}'.format(layer_idx, i, i + 1, s_layer_idx, y_layer_idx))
        if 0 < layer_idx < nlayers - 1:
            # all layers have two sub-layers except the first and the last
            p1 = W[i]
            p2 = W[i + 1]
            # print('layer {} norm {}'.format(layer_idx, ((p1 ** 2).sum() + (p2 ** 2).sum()).item()))
            # debug_norms.append(((p1 ** 2).sum() + (p2 ** 2).sum()).item())
            assert p1.size(0) == p2.size(1) and p1.size(1) == 1
            p2_t_s = (p2.data * p_scale[i+1] if eps is None else p2.data).transpose(0, 1).contiguous()
            p2_t_s = p2_t_s.view(p2.size(1), -1)

            p1_s = p1.data * p_scale[i] if eps is None else p1.data
            if eps is not None:
                denom1 = optimizer.state[p1]['exp_avg_sq'].sqrt().add_(eps).view(p1.size(0), -1)
                denom2 = optimizer.state[p2]['exp_avg_sq'].sqrt().add_(eps).transpose(0, 1).contiguous().view(p2.size(1), -1)
                # a = torch.sum((p2_t ** 2) * denom2, dim=1)
                a = torch.sum((p1_s.data.view(p1.size(0), -1) ** 2) * denom1, dim=1) + torch.sum((p2_t_s ** 2) * denom2, dim=1)
            else:
                # a = torch.sum(p2_t ** 2, dim=1)  # only consider the second (pointwise) conv layer is enough
                a = torch.sum(p1_s.data.view(p1.size(0), -1) ** 2, dim=1) + torch.sum(p2_t_s ** 2, dim=1)
        elif layer_idx == 0:
            # the first layer (normal conv2d)
            p = W[i + 1]
            assert p.dim() == 4 and i + 1 == 0
            # do not prune the first layer, do nothing
            continue
        else:
            # the last layer (normal fc)
            p = W[i]
            # print('layer {} norm {}'.format(layer_idx, (p ** 2).sum().item()))
            # debug_norms.append((p ** 2).sum().item())
            assert p.dim() == 2 and i == len(W) - 1
            if eps is not None:
                denom = optimizer.state[p]['exp_avg_sq'].sqrt().add_(eps)
                a = torch.sum((p.data ** 2) * denom, dim=0)
            else:
                a = torch.sum(p.data ** 2, dim=0)

        # if layer_idx == 2:
        #     print(a.sort()[0].cpu().numpy())
        # print('layer {} overall norm {}'.format(layer_idx, a.sum().item()))
        # debug_norms2.append(a.sum().item())

        if rho_y == 0.0:
            a_rho = 0.0
        else:
            _, indices = torch.sort(a, descending=True)
            a_rank = torch.zeros_like(a)
            a_rank[indices] = torch.arange(1, a.numel() + 1, device=a_rank.device, dtype=a_rank.dtype)
            a_rho = rho_y * alpha * (torch.clamp(a_rank - s_layer_idx, min=0.0) ** 2
                                     - torch.clamp(a_rank - s_layer_idx - 1, min=0.0) ** 2)

        # project y
        if y_layer_idx > 0.0:
            a_topk, _ = torch.topk(a - a_rho, k=int(math.floor(s_layer_idx)), largest=True, sorted=True)
            y_layer_idx = min(y_layer_idx, a_topk[-1].item() / (2 * alpha))
            d_model.y.data[layer_idx] = y_layer_idx

        z_idx = (a < a_rho + 2 * alpha * y_layer_idx)

        if 0 < layer_idx < nlayers - 1:
            p1.data[z_idx, :, :, :] = 0.0
            p2.data[:, z_idx, :, :] = 0.0
        elif layer_idx == 0:
            p.data[:, z_idx, :, :] = 0.0
        else:
            p.data[:, z_idx] = 0.0

    # print('1')
    # print(array1d_repr(debug_norms))
    # print('2')
    # print(array1d_repr(debug_norms2))

def admm_w_update_prox_resnet(p_model, d_model, optimizer, rho_y):
    i = 0
    assert len(optimizer.param_groups) == 1
    alpha = optimizer.param_groups[0]['lr']
    if isinstance(optimizer, torch.optim.Adam):
        eps = optimizer.param_groups[0]['eps']
    elif isinstance(optimizer, torch.optim.SGD):
        eps = None
    else:
        raise NotImplementedError
    if isinstance(p_model.net, torch.nn.DataParallel):
        p_list = p_model.net.module.get_cp_weights()
    else:
        p_list = p_model.net.get_cp_weights()

    for p in p_list:
        if p is not None:
            if eps is not None:
                denom = optimizer.state[p]['exp_avg_sq'].sqrt().add_(eps)
            else:
                denom = None
            p = p.data
            s_i = float(p_model.s.data[i])
            y_i = float(d_model.y.data[i])
            input_width = p.size(1)
            p_t = p.data.transpose(0, 1).contiguous()
            p_t = p_t.view(input_width, -1)
            if denom is not None:
                denom = denom.data.transpose(0, 1).contiguous().view(input_width, -1)
                a = torch.sum((p_t ** 2) * denom, dim=1).data
            else:
                a = torch.sum(p_t ** 2, dim=1).data

            # print('i={}, y={}, p_shape={}'.format(i, y_i, p.shape))
            if rho_y == 0.0:
                a_rho = 0.0
            else:
                _, indices = torch.sort(a, descending=True)
                a_rank = torch.zeros_like(a)
                a_rank[indices] = torch.arange(1, a.numel() + 1, device=a_rank.device, dtype=a_rank.dtype)
                a_rho = rho_y * alpha * (torch.clamp(a_rank - s_i, min=0.0) ** 2
                                         - torch.clamp(a_rank - s_i - 1, min=0.0) ** 2)

            # project y
            if y_i > 0.0:
                a_topk, _ = torch.topk(a - a_rho, k=int(math.floor(s_i)), largest=True, sorted=True)
                y_i = min(y_i, a_topk[-1].item() / (2 * alpha))
                d_model.y.data[i] = y_i
            z_idx = (a < a_rho + 2 * alpha * y_i)

            # assert p_t.size(0) >= s_i, "error: {} < {}".format(p_t.size(0), s_i)
            if p.dim() == 4:
                p.data[:, z_idx, :, :] = 0.0
            else:
                assert False
        i += 1

    assert i + 1 == p_model.s.numel(), "eror:{} != {}".format(i + 1, p_model.s_scaled.numel())

def admm_dual_projection(d_model):
    d_model.z.data.clamp_(min=0.0)
    d_model.y.data.clamp_(min=0.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model-Free Energy Constrained Training')
    parser.add_argument('--net', default='alexnet', help='network arch')
    parser.add_argument('--energymodel', default='./energymodel_alex.pkl',
                        help='energy prediction model file')

    parser.add_argument('--dataset', default='imagenet', help='dataset used in the experiment')
    parser.add_argument('--datadir', default='./ILSVRC_CLS', help='dataset dir in this machine')

    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=512, help='batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for training')

    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--plr', type=float, default=1e-5, help='primal learning rate')
    parser.add_argument('--padam', action='store_true', help='use adam for primal net update')
    parser.add_argument('--padam_beta', default='0.9,0.999', help='betas of adam for primal net update')

    parser.add_argument('--pslr', type=float, default=0.01, help='primal learning rate for sparsity')
    parser.add_argument('--psgrad_mask', action='store_true', help='update s only when s.grad < 0')
    parser.add_argument('--psgrad_clip', type=float, default=None, help='clip s.grad to')
    parser.add_argument('--l2wd', type=float, default=0.0, help='l2 weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='primal momentum (if using sgd)')
    parser.add_argument('--zinit', type=float, default=0.0, help='initial dual variable z')
    parser.add_argument('--yinit', type=float, default=0.0, help='initial dual variable y')
    parser.add_argument('--lr_decay', type=float, default=1.0, help='learning rate (default: 1)')
    parser.add_argument('--s_int', type=int, default=1, help='how many batches for updating s')

    parser.add_argument('--randinit', action='store_true', help='use random init')
    parser.add_argument('--pretrain', default=None, help='file to load pretrained model')

    parser.add_argument('--seed', type=int, default=117, help='random seed (default: 117)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test_interval', type=int, default=1, help='how many epochs to wait before another test')
    parser.add_argument('--save_interval', type=int, default=-1, help='how many epochs to wait before save a model')
    parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
    parser.add_argument('--distill', type=float, default=0.5, help='distill loss weight')
    parser.add_argument('--rho_z', type=float, default=1.0, help='ADMM hyperparameter: rho for z')
    parser.add_argument('--rho_y', type=float, default=1.0, help='ADMM hyperparameter: rho for y')

    parser.add_argument('--budget', type=float, default=0.00, help='energy budget')

    parser.add_argument('--dadam', action='store_true', help='use adam for dual')
    parser.add_argument('--mgpu', action='store_true', help='enable using multiple gpus')
    parser.add_argument('--slb', type=float, default=None, help='sparsity lower bound')
    parser.add_argument('--eval', action='store_true', help='eval in the begining')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
    # rm old contents in dir
    print('remove old contents in {}'.format(args.logdir))
    os.system('rm -rf ' + args.logdir)

    # create log file
    misc.logger.init(args.logdir, 'train_log')
    print = misc.logger.info

    # backup the src
    os.system('zip -qj ' + os.path.join(args.logdir, 'src.zip') + ' {}/*.py'.format(
        os.path.dirname(os.path.realpath(__file__))))

    print('command:\npython {}'.format(' '.join(sys.argv)))
    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")

    # set up random seeds
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # get training and validation data loaders
    normalize = None
    class_offset = 0
    performance_eval = eval_loss_acc1_acc5
    if args.net == 'mobilenet-imagenet':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        class_offset = 1
    if args.dataset == 'cityscapes':
        performance_eval = eval_loss_iou

    tr_loader, val_loader, train_loader4eval = get_data_loaders(data_dir=args.datadir,
                                                                dataset=args.dataset,
                                                                batch_size=args.batch_size,
                                                                val_batch_size=args.val_batch_size,
                                                                num_workers=args.num_workers,
                                                                normalize=normalize)
    # get network model
    model, teacher_model = get_net_model(net=args.net, pretrained_dataset=args.dataset, dropout=False,
                                         pretrained=not args.randinit)

    # pretrained model
    if args.pretrain is not None and os.path.isfile(args.pretrain):
        print('load pretrained model:{}'.format(args.pretrain))
        model.load_state_dict(torch.load(args.pretrain))
    elif args.pretrain is not None:
        print('fail to load pretrained model: {}'.format(args.pretrain))

    # set up multi-gpus
    if args.mgpu:
        assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1
        model = torch.nn.DataParallel(model)
        teacher_model = torch.nn.DataParallel(teacher_model)

    netl2wd = args.l2wd
    l2wd_loss = lambda m: 0.0
    # ADMM model wrappers
    # energy estimator
    if args.net.startswith('alexnet'):
        primal_model = EnergyConstrainedADMM_P(model, Alexnet_width_ub)
        dual_model = EnergyConstrainedADMM_D(len(Alexnet_width_ub), args.zinit, args.yinit)
        if '_nn' in args.energymodel:
            energy_estimator = EnergyEstimateNet(n_nodes=[len(Alexnet_width_ub) - 1, 64, 1],
                                      preprocessor=EnergyEstimateWidthRescale(scales=(Alexnet_width_ub)))
        else:
            energy_estimator = EnergyEstimateNet(n_nodes=[len(Alexnet_width_ub) - 1, 1],
                                      preprocessor=EnergyEstimateWidthRescale(scales=(Alexnet_width_ub)))

        width_ub = Alexnet_width_ub
        column_sparsity = column_sparsity_common
        admm_w_update_prox = admm_w_update_prox_common
    elif args.net == 'erfnet-cityscapes':
        width_ub = [3] + erfnet().get_cpwub() + [20]
        print(width_ub)
        primal_model = EnergyConstrainedADMM_P(model, width_ub)
        dual_model = EnergyConstrainedADMM_D(len(width_ub), args.zinit, args.yinit)
        if '_nn' in args.energymodel:
            energy_estimator = EnergyEstimateNet(n_nodes=[len(width_ub) - 1, 64, 1],
                                      preprocessor=EnergyEstimateWidthRescale(scales=(width_ub)))
        else:
            energy_estimator = EnergyEstimateNet(n_nodes=[len(width_ub) - 1, 1],
                                      preprocessor=EnergyEstimateWidthRescale(scales=(width_ub)))
        column_sparsity = column_sparsity_resnet
        admm_w_update_prox = admm_w_update_prox_resnet
    elif args.net.startswith('mobilenet'):
        primal_model = EnergyConstrainedADMM_P(model, Mobilenet_width_ub)
        dual_model = EnergyConstrainedADMM_D(len(Mobilenet_width_ub), args.zinit, args.yinit)
        if '_nn' in args.energymodel:
            energy_estimator = EnergyEstimateNet(n_nodes=[len(Mobilenet_width_ub) - 1, 64, 1],
                                      preprocessor=EnergyEstimateWidthRescale(scales=(Mobilenet_width_ub)))
        else:
            energy_estimator = EnergyEstimateNet(n_nodes=[len(Mobilenet_width_ub) - 1, 1],
                                      preprocessor=EnergyEstimateWidthRescale(scales=(Mobilenet_width_ub)))
        width_ub = Mobilenet_width_ub
        column_sparsity = lambda m, out=None: column_sparsity_mbnet(m, out=out, verbose=False, zero_pre=True)
        admm_w_update_prox = admm_w_update_prox_mbnet
        l2wd_loss = lambda m: m.l2wd_loss(netl2wd) if not isinstance(m, torch.nn.DataParallel) else m.module.l2wd_loss(netl2wd)
        netl2wd = 0.0
    else:
        raise NotImplementedError

    energy_estimator.load_state_dict(torch.load(args.energymodel))
    for p in energy_estimator.parameters():
        p.requires_grad = False

    if args.cuda:
        if args.distill > 0.0:
            teacher_model.cuda()
        primal_model.cuda()
        dual_model.cuda()
        energy_estimator.cuda()

    # sparsity lower bound
    if args.slb is not None:
        slb = primal_model.s.data * args.slb
    else:
        slb = torch.ones_like(primal_model.s.data)

    # Optimizers
    args.zlr = args.rho_z
    args.ylr = args.rho_y

    primal_optimizer_list = []
    if args.padam:
        betas = (float(args.padam_beta.split(',')[0]), float(args.padam_beta.split(',')[1]))
        primal_optimizer_list.append(torch.optim.Adam(primal_model.net.parameters(), lr=args.plr, betas=betas,
                                                      weight_decay=netl2wd))
    else:
        primal_optimizer_list.append(torch.optim.SGD(primal_model.net.parameters(), lr=args.plr, momentum=args.momentum,
                                                     weight_decay=netl2wd))

    primal_optimizer_list.append(torch.optim.SGD([primal_model.s], lr=args.pslr))

    if args.dadam:
        dual_optimizer = torch.optim.Adam(dual_model.get_param_dicts(zlr=args.zlr, ylr=args.ylr), lr=1e-3)
    else:
        dual_optimizer = torch.optim.SGD(dual_model.get_param_dicts(zlr=args.zlr, ylr=args.ylr), lr=1e-2)

    loss_func = lambda m, x, y: joint_loss(model=m, data=x, target=y, teacher_model=teacher_model, distill=args.distill) \
                                + l2wd_loss(m)

    if args.dataset == 'cityscapes':
        loss_func = lambda m, x, y: joint_loss_cityscape(model=m, data=x, target=y, teacher_model=teacher_model, distill=args.distill) \
                                + l2wd_loss(m)

    plot_data = PlotData()
    if args.eval or args.dataset != 'imagenet':
        val_loss, val_acc1, val_acc5 = performance_eval(primal_model.net, val_loader, loss_func, args.cuda,
                                                           class_offset=class_offset)
        print('**Validation loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}'.format(val_loss, val_acc1,
                                                                                              val_acc5))
    else:
        val_acc1 = 0.0
        print('For imagenet, skip the first validation evaluation.')

    best_acc = val_acc1

    old_file = None
    energy_residue_ub = energy_estimator(primal_model.s).item() - args.budget
    s_real = primal_model.s.data.clone()
    # initialize s
    column_sparsity(primal_model.net, out=s_real.data)
    primal_model.s.data.copy_(s_real)

    t_begin = time.time()
    log_tic = t_begin
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(tr_loader):
            primal_model.net.train()
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            w_loss = loss_func(primal_model.net, data, target + class_offset)
            # update network weights
            primal_optimizer_list[0].zero_grad()
            w_loss.backward()
            primal_optimizer_list[0].step()
            admm_w_update_prox(primal_model, dual_model, primal_optimizer_list[0], args.rho_y)
            column_sparsity(primal_model.net, out=s_real.data)
            # evaluate current energy
            cur_energy = energy_estimator(s_real).item()
            # update sparsity variables s

            if batch_idx % args.log_interval == 0:
                print('updating s...')
            for _ in range(args.s_int):
                energy_residue = energy_estimator(primal_model.s) - args.budget
                s_residue = s_real - primal_model.s

                # compute lr for s
                pslr = args.pslr

                s_loss_rest = 0.5 * args.rho_z * torch.clamp(energy_residue, min=0.0) ** 2 \
                              + 0.5 * args.rho_y * torch.sum(torch.clamp(s_residue, min=0.0)) ** 2 \
                              + torch.sum(dual_model.y * s_residue)
                s_grad1 = torch.autograd.grad(s_loss_rest, primal_model.s, only_inputs=True, retain_graph=True)[0].data
                # print(s_grad1)
                s_grad2 = torch.autograd.grad(energy_residue, primal_model.s, only_inputs=True, retain_graph=True)[0].data
                # print(s_grad2)
                s_update_mask = primal_model.s > slb
                z_min = ((-s_grad1.clamp(max=0.0) + 1e-3) / s_grad2.clamp(min=0.0))[s_update_mask].min().item()

                dual_model.z.data.clamp_(min=z_min)
                s_loss = s_loss_rest + dual_model.z * energy_residue
                if batch_idx % args.log_interval == 0:
                    print('s_loss={:.8e}'.format(s_loss.item()))
                primal_optimizer_list[1].zero_grad()
                # s_loss.backward()
                if primal_model.s.grad is None:
                    primal_model.s.grad = s_grad1 + dual_model.z.data * s_grad2
                else:
                    primal_model.s.grad.data.copy_(s_grad1 + dual_model.z.data * s_grad2)

                if args.psgrad_mask:
                    primal_model.s.grad.data.clamp_(min=0.0)

                primal_model.s.grad[s_update_mask == 0] = 0.0
                # s grad cliping
                if args.psgrad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(primal_model.s, args.psgrad_clip)

                primal_optimizer_list[1].step()
                # s projection
                primal_model.s.data.clamp_(min=1.0)

            if batch_idx % args.log_interval == 0:
                print('======================================================')
                print('+-------------- epoch {}, batch {}/{} ----------------+'.format(epoch, batch_idx,
                                                                                       len(tr_loader)))
                log_toc = time.time()
                print('primal update: net loss={:.4e}, pslr={:.4e}, current energy={:.4e}, time_elapsed={:.3f}s'.format(
                    w_loss.item(), pslr, cur_energy, log_toc - log_tic))
                log_tic = time.time()

                print('s_var ={}'.format(array1d_repr(primal_model.s.data.view(-1).cpu().numpy())))
                # print(primal_model.s.grad.data.view(-1).cpu().numpy())
                print('s_real={}'.format(array1d_repr(s_real.data.view(-1).cpu().numpy())))
                print('s.grad={}'.format(array1d_repr(primal_model.s.grad.data.view(-1).cpu().numpy())))
                print('+-----------------------------------------------------+')

            # dual update
            energy_residue = (energy_estimator(primal_model.s) - args.budget).detach()
            s_residue = (s_real - torch.floor(primal_model.s)).detach()
            dual_optimizer.zero_grad()
            (-dual_model.z * energy_residue - torch.sum(dual_model.y * s_residue)).backward()

            dual_optimizer.step()
            admm_dual_projection(dual_model)
            # dual_model.y.data[dual_model.y.grad.data > 0.0] = 0.0

            if batch_idx % args.log_interval == 0:
                print('dual update: z={:.4e}, zlr={:.4e}, ylr={:.4e}'
                      .format(dual_model.z.item(), dual_optimizer.param_groups[0]['lr'],
                              dual_optimizer.param_groups[1]['lr']))
                print('y.data={}'.format(array1d_repr(dual_model.y.data.view(-1).cpu().numpy())))
                print('y.grad={}'.format(array1d_repr(dual_model.y.grad.data.view(-1).cpu().numpy())))

                print('+-----------------------END---------------------------+')
                print('======================================================')

            # Stop
            if cur_energy <= args.budget:
                break

        # decay lr
        if epoch > 0 and epoch % 30 == 0:
            for param_group in primal_optimizer_list[0].param_groups:
                param_group['lr'] *= args.lr_decay

        if epoch % args.test_interval == 0:
            plot_data.append('energy', cur_energy)
            val_loss, val_acc1, val_acc5 = performance_eval(primal_model.net, val_loader, loss_func, args.cuda,
                                                               class_offset=class_offset)
            plot_data.append('val_loss', val_loss)
            plot_data.append('val_acc1', val_acc1)

            # also evaluate training data
            tr_loss, tr_acc1, tr_acc5 = performance_eval(primal_model.net, train_loader4eval, loss_func,
                                                            args.cuda, class_offset=class_offset)
            print('###Training loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}'.format(tr_loss, tr_acc1,
                                                                                                 tr_acc5))
            plot_data.append('tr_loss', tr_loss)
            plot_data.append('tr_acc1', tr_acc1)

            print(
                '***Validation loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}, current energy:{:.4e}'.format(
                    val_loss, val_acc1,
                    val_acc5, cur_energy))
            # save current model
            model_snapshot(primal_model, os.path.join(args.logdir, 'primal_model_latest.pkl'))
            plot_data.dump(os.path.join(args.logdir, 'plot_data.pkl'))

        if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
            model_snapshot(primal_model, os.path.join(args.logdir, 'primal_model_epoch{}.pkl'.format(epoch)))

        elapse_time = time.time() - t_begin
        reduced_energy = energy_residue_ub - (cur_energy - args.budget)
        if reduced_energy <= 0:
            print('No energy reduced! Considering improve rho_z.')
        else:
            speed_energy = elapse_time / reduced_energy
            eta = speed_energy * energy_residue_ub - elapse_time
            print("Elapsed {:.2f}s, ets {:.2f}s".format(elapse_time, eta))

        if cur_energy <= args.budget:
            print('Constraints satisfied! current energy={:.4e}, Stop.'.format(cur_energy))
            break
