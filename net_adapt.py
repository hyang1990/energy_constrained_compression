import argparse
import copy
import datetime
import math
import os
import random
import sys
import time

import numpy as np
import torch
from energy_estimator import EnergyEstimateNet, EnergyEstimateWidthRescale, Mobilenet_width_ub, \
    Alexnet_width_ub
from erfnet_cp import erfnet
from utils import get_data_loaders, joint_loss, PlotData, \
    eval_loss_acc1_acc5, column_sparsity_common, argmax, class_balance_holdout, \
    get_net_model, column_sparsity_mbnet, eval_loss_iou, column_sparsity_resnet, joint_loss_cityscape, \
    filter_projection_resnet, simple_random_holdout, fill_model_weights, model_mask, maskproj
from torchvision import transforms
import misc


def model_based_energy_estimator(model_width, energy_predictor):
    # model-free estimation
    return energy_predictor(model_width)


def choose_num_filters(cur_layer_width, energy_model, layer_idx, budget):
    assert 0 <= layer_idx < len(cur_layer_width)
    layer_width = copy.deepcopy(cur_layer_width)
    energy = energy_model(layer_width)
    while energy > budget:
        if layer_width[layer_idx] > 1:
            layer_width[layer_idx] -= 1
            energy = energy_model(layer_width)
        else:
            return None

    return int(layer_width[layer_idx])


def filter_projection_common(model, layer_idx, num_filters):
    i = 0
    for name, p in model.named_parameters():
        if name.endswith('weight') and p.dim() > 1:
            if i == layer_idx:
                input_width = p.size(1)
                if input_width != last_output_width:
                    assert input_width > last_output_width
                    assert input_width % last_output_width == 0
                    input_width = last_output_width
                if input_width == num_filters:
                    return
                p_t = p.data.transpose(0, 1).contiguous()
                p_t_old_shape = p_t.shape
                p_t = p_t.view(input_width, -1)
                a = torch.sum(p_t ** 2, dim=1)
                _, indices = torch.topk(a, input_width - num_filters, largest=False, sorted=False)
                if p.dim() == 4:
                    p.data[:, indices, :, :] = 0.0
                else:
                    if input_width == p.size(1):
                        p.data[:, indices] = 0.0
                    else:
                        p_t[indices, :] = 0.0
                        p.data.copy_(p_t.view(*p_t_old_shape).transpose(0, 1))
                return model
            last_output_width = p.size(0)
            i += 1


def filter_projection_mbnet(model, layer_idx, num_filters):
    # the first layer is normal conv, the last layer is fc, grouped conv are in the middle
    nlayers = len(Mobilenet_width_ub) - 1
    W = []
    for name, p in model.named_parameters():
        if name.endswith('weight') and p.dim() > 1:
            W.append(p)

    i = layer_idx * 2 - 1
    if 0 < layer_idx < nlayers - 1:
        # all layers have two sub-layers except the first and the last
        p1 = W[i]
        p2 = W[i + 1]
        assert p1.size(0) == p2.size(1) and p1.size(1) == 1
        input_width = p2.size(1)
        p2_t_s = p2.data.transpose(0, 1).contiguous()
        p2_t_s = p2_t_s.view(p2.size(1), -1)

        p1_s = p1.data
        a = torch.sum(p1_s.data.view(p1.size(0), -1) ** 2, dim=1) + torch.sum(p2_t_s ** 2, dim=1)
    elif layer_idx == 0:
        # the first layer (normal conv2d)
        raise Warning('the first input layer should not be pruned')
        p = W[0]
    else:
        # the last layer (normal fc)
        p = W[i]
        assert p.dim() == 2 and i == len(W) - 1
        a = torch.sum(p.data ** 2, dim=0)
        input_width = p.size(1)

    if input_width == num_filters:
        return
    _, z_idx = torch.topk(a, input_width - num_filters, largest=False, sorted=False)

    if 0 < layer_idx < nlayers - 1:
        p1.data[z_idx, :, :, :] = 0.0
        p2.data[:, z_idx, :, :] = 0.0
    elif layer_idx == 0:
        p.data[:, z_idx, :, :] = 0.0
    else:
        p.data[:, z_idx] = 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NetAdapt Implementation')
    parser.add_argument('--net', default='alexnet', help='network arch')
    parser.add_argument('--budget', type=float, default=0.2, help='energy budget')
    parser.add_argument('--abs_budget', action='store_true', help='absolute budget')
    parser.add_argument('--bdecay', type=float, default=0.96, help='budget decay rate')
    parser.add_argument('--dataset', default='imagenet', help='dataset used in the experiment')
    parser.add_argument('--datadir', default='/home/hyang/ssd2/ILSVRC_CLS', help='dataset dir in this machine')
    parser.add_argument('--nclasses', type=int, default=None, help='number of classes for dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=512, help='batch size for evaluation')
    parser.add_argument('--energymodel', required=True, help='energy predictor model')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for train')
    parser.add_argument('--lr', type=float, default=1e-3, help='primal learning rate')
    parser.add_argument('--l2wd', type=float, default=1e-4, help='l2 weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='primal momentum')
    parser.add_argument('--pretrain', default=None, help='file to load pretrained model')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--lt_epochs', type=int, default=0, help='number of epochs to train')
    parser.add_argument('--seed', type=int, default=117, help='random seed (default: 117)')
    parser.add_argument('--randinit', action='store_true', help='use random init')
    parser.add_argument('--eval', action='store_true', help='eval mode')
    parser.add_argument('--finetune', action='store_true', help='finetune mode')
    parser.add_argument('--optim', default='sgd', help='optimizer')

    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test_interval', type=int, default=1, help='how many epochs to wait before another test')
    parser.add_argument('--save_interval', type=int, default=-1, help='how many epochs to wait before save a model')
    parser.add_argument('--logdir', default=None, help='folder to save to the log')
    parser.add_argument('--distill', type=float, default=0.5, help='distill loss weight')
    parser.add_argument('--mgpu', action='store_true', help='enable using multiple gpus')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    if args.logdir is None:
        args.logdir = 'log/' + sys.argv[0] + str(datetime.datetime.now().strftime("_%Y_%m_%d_AT_%H_%M_%S"))

    args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
    # rm old contents in dir
    print('remove old contents in {}'.format(args.logdir))
    os.system('rm -rf ' + args.logdir)
    # create log file
    misc.logger.init(args.logdir, 'train_log')
    print = misc.logger.info
    # backup the src
    os.system('zip -qj ' + os.path.join(args.logdir, 'src.zip') + ' ./*.py')

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

    # prepare data loaders
    normalize = None
    class_offset = 0
    performance_eval = eval_loss_acc1_acc5
    if args.net == 'mobilenet-imagenet':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        class_offset = 1
    if args.dataset == 'cityscapes':
        performance_eval = eval_loss_iou
    # get training and validation data loaders
    tr_loader, val_loader, train_loader4eval = get_data_loaders(data_dir=args.datadir,
                                                                dataset=args.dataset,
                                                                batch_size=args.batch_size,
                                                                val_batch_size=args.val_batch_size,
                                                                num_workers=args.num_workers,
                                                                normalize=normalize)
    if args.nclasses is None:
        if args.dataset == 'imagenet':
            args.nclasses = 1000
        elif args.dataset.startswith('cifar100'):
            args.nclasses = 100
        elif args.dataset.startswith('mnist'):
            args.nclasses = 10
        elif args.dataset == 'cityscapes':
            args.nclasses = 19
        else:
            raise ValueError('You mush pass nclasses for this dataset')

    old_tr_loader = tr_loader
    if not args.finetune:
        if args.dataset == 'cityscapes':
            tr_loader, holdout_loader = simple_random_holdout(tr_loader.dataset, n_classes=args.nclasses,
                                                              batch_size=tr_loader.batch_size,
                                                              num_workers=tr_loader.num_workers, n_sample4class=10)
        else:
            tr_loader, holdout_loader = class_balance_holdout(tr_loader.dataset, n_classes=args.nclasses,
                                                              batch_size=tr_loader.batch_size,
                                                              num_workers=tr_loader.num_workers, n_sample4class=10)
        #
        # if args.dataset != 'imagenet':
        #     tr_loader, houldout_loader = class_balance_holdout(tr_loader.dataset, n_classes=args.nclasses,
        #                                                        batch_size=tr_loader.batch_size,
        #                                                        num_workers=tr_loader.num_workers, n_sample4class=10)
        # else:
        #     houldout_loader = class_balance_holdout_loader(tr_loader.dataset, n_classes=args.nclasses,
        #                                                    batch_size=args.val_batch_size,
        #                                                    num_workers=args.num_workers, n_sample4class=10)
    # get network model
    model, teacher_model = get_net_model(net=args.net, pretrained_dataset=args.dataset, dropout=False,
                                         pretrained=not args.randinit)

    # pretrained model
    if args.pretrain is not None and os.path.isfile(args.pretrain):
        print('load pretrained model:{}'.format(args.pretrain))
        model.load_state_dict(torch.load(args.pretrain))
    elif args.pretrain is not None:
        print('fail to load pretrained model: {}'.format(args.pretrain))

    netl2wd = args.l2wd
    l2wd_loss = lambda m: 0.0

    # for energy estimate
    print('================model energy summary================')
    # energy estimator
    if args.net == 'mobilenet-imagenet':
        width_ub = Mobilenet_width_ub
        filter_projection = filter_projection_mbnet
        column_sparsity = column_sparsity_mbnet
        l2wd_loss = lambda m: m.l2wd_loss(netl2wd) if not isinstance(m, torch.nn.DataParallel) else m.module.l2wd_loss(netl2wd)
        netl2wd = 0.0
    elif args.net == 'alexnet':
        width_ub = Alexnet_width_ub
        filter_projection = filter_projection_common
        column_sparsity = column_sparsity_common
    elif args.net == 'erfnet-cityscapes':
        width_ub = [3] + erfnet().get_cpwub() + [20]
        column_sparsity = column_sparsity_resnet
        filter_projection = filter_projection_resnet
    else:
        raise NotImplementedError
    if '_nn' in args.energymodel:
        energy_estimator_net = EnergyEstimateNet(n_nodes=[len(width_ub) - 1, 64, 1],
                                             preprocessor=EnergyEstimateWidthRescale(scales=(width_ub)))
    else:
        energy_estimator_net = EnergyEstimateNet(n_nodes=[len(width_ub) - 1, 1],
                                             preprocessor=EnergyEstimateWidthRescale(scales=(width_ub)))

        energy_estimator_net.load_state_dict(torch.load(args.energymodel))
    for p in energy_estimator_net.parameters():
        p.requires_grad = False
    if args.cuda:
        energy_estimator_net.cuda()

    data, target = next(iter(tr_loader))
    assert data.size(1) in [1, 3]
    args.in_channels = data.size(1)

    energy_predictor = lambda wl: energy_estimator_net(torch.tensor([args.in_channels] + wl + [width_ub[-1]],
                                                                dtype=torch.float32,
                                                                device=next(energy_estimator_net.parameters()).device)).item()

    energy_estimator = lambda m: energy_predictor(column_sparsity(m)[1:])
    energy_estimator4width = lambda m_width: energy_predictor(m_width)

    dense_model = fill_model_weights(copy.deepcopy(model), 1.0)
    budget_ub = energy_estimator(dense_model)
    del dense_model
    cur_energy = energy_estimator(model)
    print('energy on dense DNN:{:.4e}'.format(budget_ub))
    print('energy on current DNN:{:.4e}, normalized={:.4e}'.format(cur_energy, cur_energy / budget_ub))
    print('====================================================')
    print('current energy {:.4e}'.format(cur_energy))

    # set up multi-gpus
    if args.mgpu:
        assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1
        teacher_model = torch.nn.DataParallel(teacher_model)
        model = torch.nn.DataParallel(model)

    if args.cuda:
        if args.distill > 0.0:
            teacher_model.cuda()
        model.cuda()

    loss_func = lambda m, x, y: joint_loss(model=m, data=x, target=y, teacher_model=teacher_model, distill=args.distill) + l2wd_loss(m)

    if args.dataset == 'cityscapes':
        loss_func = lambda m, x, y: joint_loss_cityscape(model=m, data=x, target=y, teacher_model=teacher_model, distill=args.distill) \
                                + l2wd_loss(m)
    plot_data = PlotData()

    if args.eval or args.dataset != 'imagenet':
        val_loss, val_acc1, val_acc5 = performance_eval(model, val_loader, loss_func, args.cuda,
                                                           class_offset=class_offset)
        print('**Validation loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}'.format(val_loss, val_acc1,
                                                                                              val_acc5))
        # # also evaluate training data
        # tr_loss, tr_acc1, tr_acc5 = performance_eval(model, train_loader4eval, loss_func, args.cuda,
        #                                                 class_offset=class_offset)
        # print('###Training loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}'.format(tr_loss, tr_acc1, tr_acc5))
    else:
        val_acc1 = 0.0
        print('For imagenet, skip the first validation evaluation.')

    layer_sparsity = column_sparsity(model)[1:]
    n_layers = len(layer_sparsity)

    if not args.abs_budget:
        args.budget = args.budget * budget_ub
    cur_budget = cur_energy * args.bdecay
    n_nodes_lb = [layer_sparsity[i] for i in range(n_layers)]
    if not args.finetune:
        args.bdecay_epochs = int(math.ceil(math.log(args.budget / cur_energy, args.bdecay)))
        print('need {} iterations to reduce energy {:.4e} to {:.4e}'.format(args.bdecay_epochs, cur_energy, args.budget))
        args.st_epochs = (args.epochs - args.lt_epochs) / float(args.bdecay_epochs * n_layers)
        assert args.st_epochs > args.batch_size / float(len(tr_loader.dataset)), 'epochs is not large enough to have valid st_epochs'
        print('st_epochs = {}'.format(args.st_epochs))

    best_model = copy.deepcopy(model)
    t_begin = time.time()
    if not args.finetune:
        for epoch in range(args.bdecay_epochs):
            model_candidates_acc = [0.0] * n_layers
            if epoch == 0:
                cur_model = copy.deepcopy(model)
                # model_candidates = [copy.deepcopy(model) for _ in range(n_layers)]
            else:
                pass
                # for m in model_candidates:
                #     m.load_state_dict(model.state_dict())
            if cur_energy <= args.budget:
                print('Constraints satisfied! current energy={:.4e}, Stop.'.format(cur_energy))
                break
            # for k, model_k in enumerate(model_candidates):
            for k in range(n_layers):
                cur_model.load_state_dict(model.state_dict())
                model_k = cur_model
                num_filters = choose_num_filters(layer_sparsity, energy_estimator4width, k, cur_budget)
                if num_filters is None:
                    print('cannot reduce energy to the budget by pruning channels in {}-th layer.'.format(k+1))
                    model_candidates_acc[k] = 0.0
                    continue
                else:
                    print('try to reduce num_filters from {} to {}.'.format(layer_sparsity[k], num_filters))
                filter_projection(model_k, k + 1, num_filters)
                n_nodes_lb[k] = num_filters
                # short-term fine-tune
                print('+-----------------------BEGIN---------------------------+')
                print('short term fine-tune on the result after pruning the {}-th layer.'.format(k+1))
                mask = model_mask(model_k, param_name='weight')
                if args.optim == 'sgd':
                    optimizer = torch.optim.SGD(model_k.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=netl2wd)
                elif args.optim == 'rms':
                    optimizer = torch.optim.RMSprop(model_k.parameters(), lr=0.0045, momentum=args.momentum, weight_decay=netl2wd)
                elif args.optim == 'adam':
                    optimizer = torch.optim.Adam(model_k.parameters(), lr=args.lr, weight_decay=netl2wd)
                else:
                    raise NotImplementedError
                for st_epoch in range(int(math.ceil(args.st_epochs))):
                    for batch_idx, (data, target) in enumerate(tr_loader):
                        model_k.train()
                        if st_epoch == math.ceil(args.st_epochs) - 1 and math.ceil(args.st_epochs) > args.st_epochs\
                                and batch_idx / float(len(tr_loader)) >= args.st_epochs - math.floor(args.st_epochs):
                            break
                        if args.cuda:
                            data, target = data.cuda(), target.cuda()
                        loss = loss_func(model_k, data, target + class_offset)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        maskproj(model_k, mask, param_name='weight')
                        if batch_idx % args.log_interval == 0:
                            print('+-------------- epoch {}, batch {}/{} ----------------+'.format(st_epoch, batch_idx,
                                                                                                   len(tr_loader)))
                            print('loss={:.4e}'.format(loss.item()))
                            print('+-----------------------------------------------------+')

                _, acc1, acc5 = performance_eval(model_k, holdout_loader, None, args.cuda, class_offset=class_offset)
                print('holdout acc={:.5f}, cur_budget={:4e}'.format(acc1, cur_budget))
                if acc1 > max(model_candidates_acc):
                    best_model.load_state_dict(model_k.state_dict())
                model_candidates_acc[k] = float(acc1)
                if acc1 >= 1.0:
                    print('full accuracy on holdout set, early stop')
                    break
                # loss, acc1, acc5 = performance_eval(model_k, tr_loader, loss_func, args.cuda)
                # print('training loss={:.4e}, acc1={:.5f}, acc5={:.5f}, cur_budget={:5e}'.format(loss, acc1, acc5, cur_budget))
                print('+------------------------END----------------------------+')
                print('======================================================')

            # pick the one with highest acc
            print('accuracies of candidate models:{}'.format(model_candidates_acc))
            if max(model_candidates_acc) > 0.0:
                model.load_state_dict(best_model.state_dict())
                idx = argmax(model_candidates_acc)
                # model.load_state_dict(model_candidates[idx].state_dict())
                layer_sparsity[idx] = n_nodes_lb[idx]
            else:
                print('cannot keep pruning filters! Stop.')
                break
            cur_budget *= args.bdecay

            # evaluate the current model
            loss, acc1, acc5 = performance_eval(model, val_loader, loss_func, args.cuda, class_offset=class_offset)
            cur_energy = energy_estimator(model)
            plot_data.append('val_loss', loss)
            plot_data.append('val_acc1', acc1)
            plot_data.append('val_acc5', acc5)
            plot_data.append('energy', cur_energy)
            print('=======================================================')
            print('epoch {}, current net width: {}'.format(epoch, layer_sparsity))
            print('validation loss={:.4e}, acc1={:.5f}, acc5={:.5f}, current normalized energy:{:.4e}'
                  .format(loss, acc1, acc5, cur_energy / budget_ub))

            loss, acc1, acc5 = performance_eval(model, train_loader4eval, loss_func, args.cuda, class_offset=class_offset)
            plot_data.append('tr_loss', loss)
            plot_data.append('tr_acc1', acc1)
            plot_data.append('tr_acc5', acc5)

            plot_data.dump(os.path.join(args.logdir, 'plot_data.pkl'))
            print('training loss={:.4e}, acc1={:.5f}, acc5={:.5f}, current energy: {:.4e}, current normalized energy:{:.4e}'
                  .format(loss, acc1, acc5, cur_energy, cur_energy / budget_ub))
            elapse_time = time.time() - t_begin
            speed_epoch = elapse_time / (epoch + 1)
            eta = speed_epoch * args.bdecay_epochs - elapse_time
            print("Elapsed {:.2f}s, {:.2f} s/epoch, ets {:.2f}s".format(elapse_time, speed_epoch, eta))
            print('=======================================================')

            if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
                misc.model_snapshot(model, os.path.join(args.logdir, 'primal_model_epoch{}.pkl'.format(epoch)))

            misc.model_snapshot(model, os.path.join(args.logdir, 'primal_model_latest.pkl'))

    if args.lt_epochs > 0:
        # long-term fine-tune
        mask = model_mask(model, param_name='weight')
        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=netl2wd)
        elif args.optim == 'rms':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=0.045, momentum=args.momentum, weight_decay=netl2wd)
        elif args.optim == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=netl2wd)
        else:
            raise NotImplementedError
    for lt_epoch in range(args.lt_epochs):
        for batch_idx, (data, target) in enumerate(old_tr_loader):
            model.train()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            loss = loss_func(model, data, target + class_offset)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            maskproj(model, mask, param_name='weight')
        print('long term fine-tuning... epoch: {}/{}'.format(lt_epoch, args.lt_epochs))
        val_loss, val_acc1, val_acc5 = performance_eval(model, val_loader, loss_func, args.cuda,
                                                           class_offset=class_offset)
        print(
            '***Validation loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}, current normalized energy:{:.4e}'.format(
                val_loss, val_acc1,
                val_acc5, cur_energy / budget_ub))

        misc.model_snapshot(model, os.path.join(args.logdir, 'primal_model_latest.pkl'))

    loss, acc1, acc5 = performance_eval(model, val_loader, loss_func, args.cuda, class_offset=class_offset)
    cur_energy = energy_estimator(model)

    val_loss, val_acc1, val_acc5 = performance_eval(model, val_loader, loss_func, args.cuda,
                                                       class_offset=class_offset)
    plot_data.append('val_loss', val_loss)
    plot_data.append('val_acc1', val_acc1)
    plot_data.append('val_acc5', val_acc5)

    # also evaluate training data
    tr_loss, tr_acc1, tr_acc5 = performance_eval(model, train_loader4eval, loss_func,
                                                    args.cuda, class_offset=class_offset)
    print('###Training loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}'.format(tr_loss, tr_acc1,
                                                                                         tr_acc5))
    plot_data.append('tr_loss', tr_loss)
    plot_data.append('tr_acc1', tr_acc1)
    plot_data.append('tr_acc5', tr_acc5)

    print(
        '***Validation loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}, current normalized energy:{:.4e}'.format(
            val_loss, val_acc1,
            val_acc5, cur_energy / budget_ub))
