import copy
import numpy as np
import random
import os
import pickle
import warnings

import torch
from cityscapes import MyCoTransform, cityscapes
from erfnet_cp import erfnet
from torch import nn

from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from torchvision import datasets, transforms
from pt_models import myalexnet
from torchvision.models import alexnet

import torch.nn.functional as F
from energy_estimator import Mobilenet_width_ub
from mobilenet import MobileNet


def save_obj(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


class PlotData(object):
    def __init__(self):
        self.data = {}

    def append(self, name, number):
        if name in self.data:
            self.data[name].append(float(number))
        else:
            self.data[name] = [float(number)]

    def dump(self, filepath):
        save_obj(self.data, filepath)


class SubsetSequentialSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def simple_random_holdout(train_dataset, n_classes, batch_size, num_workers, n_sample4class=10):
    nsample = n_classes * n_sample4class
    rand_idx = torch.randperm(len(train_dataset)).tolist()
    holdout_idx = rand_idx[:nsample]
    rand_idx = rand_idx[nsample:]
    train_sampler = SubsetRandomSampler(rand_idx)
    holdout_sampler = SubsetSequentialSampler(holdout_idx)
    return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                       pin_memory=True, sampler=train_sampler), \
           torch.utils.data.DataLoader(
               train_dataset,
               batch_size=batch_size,
               shuffle=False,
               num_workers=num_workers,
               pin_memory=True,
               sampler=holdout_sampler)


def class_balance_holdout(train_dataset, n_classes, batch_size, num_workers, n_sample4class=10):
    rand_idx = torch.randperm(len(train_dataset)).tolist()
    n_sampled = [0] * n_classes
    tr_idx = []
    holdout_idx = []
    offset = None
    for i, idx in enumerate(rand_idx):
        data, label = train_dataset[idx]
        if n_sampled[label] < n_sample4class:
            # holdout_idx.append(rand_idx.pop(i))
            holdout_idx.append(rand_idx[i])
            n_sampled[label] += 1
        else:
            tr_idx.append(rand_idx[i])
        if min(n_sampled) == n_sample4class:
            offset = i + 1
            break

    train_sampler = SubsetRandomSampler(tr_idx + rand_idx[offset:])
    holdout_sampler = SubsetRandomSampler(holdout_idx)

    return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                       sampler=train_sampler), torch.utils.data.DataLoader(train_dataset,
                                                                                           batch_size=batch_size,
                                                                                           shuffle=False,
                                                                                           num_workers=num_workers,
                                                                                           sampler=holdout_sampler)


def class_balance_holdout_loader(train_dataset, n_classes, batch_size, num_workers, n_sample4class=10):
    N = len(train_dataset)
    holdout_dataset = copy.deepcopy(train_dataset)
    holdout_dataset.samples = []
    n_sampled = [0] * n_classes
    while True:
        idx = random.randint(0, len(train_dataset))
        data, label = train_dataset[idx]
        if n_sampled[label] < n_sample4class:
            holdout_dataset.samples.append(train_dataset.samples.pop(idx))
            n_sampled[label] += 1
            if min(n_sampled) == n_sample4class:
                break

    assert len(holdout_dataset) + len(train_dataset) == N and len(holdout_dataset) == n_sample4class * n_classes
    return torch.utils.data.DataLoader(holdout_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def get_data_loaders(data_dir, dataset='imagenet', batch_size=32, val_batch_size=512, num_workers=0, nsubset=-1,
                     normalize=None):
    if dataset == 'imagenet':
        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')
        if normalize is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        # train_dataset = datasets.ImageFolder(
        #     traindir,
        #     transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))

        if nsubset > 0:
            rand_idx = torch.randperm(len(train_dataset))[:nsubset]
            print('use a random subset of data:')
            print(rand_idx)
            train_sampler = SubsetRandomSampler(rand_idx)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=val_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        # use 10K training data to see the training performance
        train_loader4eval = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=val_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            sampler=SubsetRandomSampler(torch.randperm(len(train_dataset))[:10000]))

        return train_loader, val_loader, train_loader4eval
    elif dataset == 'cityscapes':
        enc = False
        co_transform = MyCoTransform(enc, augment=True, height=512)
        co_transform_val = MyCoTransform(enc, augment=False, height=512)
        dataset_train = cityscapes(data_dir, co_transform, 'train')
        dataset_train4eval = cityscapes(data_dir, co_transform_val, 'train')
        dataset_val = cityscapes(data_dir, co_transform_val, 'val')
        train_loader = torch.utils.data.DataLoader(dataset_train, num_workers=num_workers, batch_size=batch_size, shuffle=True)
        train_loader4eval = torch.utils.data.DataLoader(dataset_train4eval, num_workers=num_workers, batch_size=val_batch_size, shuffle=False)
        val_loader = torch.utils.data.DataLoader(dataset_val, num_workers=num_workers, batch_size=val_batch_size, shuffle=False)
        return train_loader, val_loader, train_loader4eval
    else:
        raise NotImplementedError

imagenet_pretrained_mbnet_path = os.path.dirname(
    os.path.realpath(__file__)) + '/pretrained/imagenet_pretrained_mbnet.pt'


def get_net_model(net='alexnet', pretrained_dataset='imagenet', dropout=False, pretrained=True):
    if net == 'alexnet':
        model = myalexnet(pretrained=(pretrained_dataset == 'imagenet') and pretrained, dropout=dropout)
        teacher_model = alexnet(pretrained=(pretrained_dataset == 'imagenet'))
    elif net == 'mobilenet-imagenet':
        model = MobileNet(num_classes=1001, dropout=dropout)
        if pretrained and pretrained_dataset == 'imagenet':
            model.load_state_dict(torch.load(imagenet_pretrained_mbnet_path))
        teacher_model = MobileNet(num_classes=1001)
        if os.path.isfile(imagenet_pretrained_mbnet_path):
            teacher_model.load_state_dict(torch.load(imagenet_pretrained_mbnet_path))
        else:
            warnings.warn('failed to import teacher model!')
    elif net == 'erfnet-cityscapes':
        model = erfnet(pretrained=(pretrained_dataset == 'cityscapes') and pretrained, num_classes=20, dropout=dropout)
        teacher_model = erfnet(pretrained=(pretrained_dataset == 'cityscapes'), num_classes=20)
    else:
        raise NotImplementedError

    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()

    return model, teacher_model


def ncorrect(output, target, topk=(1,)):
    """Computes the numebr of correct@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum().item()
        res.append(correct_k)
    return res


def eval_loss_acc1_acc5(model, data_loader, loss_func=None, cuda=True, class_offset=0):
    val_loss = 0.0
    val_acc1 = 0.0
    val_acc5 = 0.0
    num_data = 0
    with torch.no_grad():
        model.eval()
        for data, target in data_loader:
            num_data += target.size(0)
            target.data += class_offset
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            if loss_func is not None:
                val_loss += loss_func(model, data, target).item()
            # val_loss += F.cross_entropy(output, target).item()
            nc1, nc5 = ncorrect(output.data, target.data, topk=(1, 5))
            val_acc1 += nc1
            val_acc5 += nc5
            # print('acc:{}, {}'.format(nc1 / target.size(0), nc5 / target.size(0)))

    val_loss /= len(data_loader)
    val_acc1 /= num_data
    val_acc5 /= num_data

    return val_loss, val_acc1, val_acc5


class IouEval(object):
    def __init__(self, nClasses, ignoreIndex=19):
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses > ignoreIndex else -1  # if ignoreIndex is larger than nClasses, consider no ignoreIndex
        classes = self.nClasses if self.ignoreIndex == -1 else self.nClasses - 1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()
        self.reset()

    def reset(self):
        self.tp.zero_()
        self.fp.zero_()
        self.fn.zero_()

    def addBatch(self, x, y):  # x=preds, y=targets
        # sizes should be "batch_size x nClasses x H x W"

        # print ("X is cuda: ", x.is_cuda)
        # print ("Y is cuda: ", y.is_cuda)

        if (x.is_cuda or y.is_cuda):
            x = x.cuda()
            y = y.cuda()

        # if size is "batch_size x 1 x H x W" scatter to onehot
        if (x.size(1) == 1):
            x_onehot = torch.zeros(x.size(0), self.nClasses, x.size(2), x.size(3))
            if x.is_cuda:
                x_onehot = x_onehot.cuda()
            x_onehot.scatter_(1, x, 1).float()
        else:
            x_onehot = x.float()

        if (y.size(1) == 1):
            y_onehot = torch.zeros(y.size(0), self.nClasses, y.size(2), y.size(3))
            if y.is_cuda:
                y_onehot = y_onehot.cuda()
            y_onehot.scatter_(1, y, 1).float()
        else:
            y_onehot = y.float()

        if (self.ignoreIndex != -1):
            ignores = y_onehot[:, self.ignoreIndex].unsqueeze(1)
            x_onehot = x_onehot[:, :self.ignoreIndex]
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores = 0

        # print(type(x_onehot))
        # print(type(y_onehot))
        # print(x_onehot.size())
        # print(y_onehot.size())

        tpmult = x_onehot * y_onehot  # times prediction and gt coincide is 1
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3,
                       keepdim=True).squeeze()
        fpmult = x_onehot * (
            1 - y_onehot - ignores)  # times prediction says its that class and gt says its not (subtracting cases when its ignore label!)
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3,
                       keepdim=True).squeeze()
        fnmult = (1 - x_onehot) * (y_onehot)  # times prediction says its not that class and gt says it is
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3,
                       keepdim=True).squeeze()

        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return torch.mean(iou).item(), iou  # returns "iou mean", "iou per class"


iou_eval = IouEval(nClasses=20)

def eval_loss_iou(model, data_loader, loss_func=None, cuda=True, class_offset=0):
    val_loss = 0.0
    num_data = 0
    iou_eval.reset()
    with torch.no_grad():
        model.eval()
        for data, target in data_loader:
            num_data += target.size(0)
            target.data += class_offset
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            if loss_func is not None:
                val_loss += loss_func(model, data, target).item()

            iou_eval.addBatch(output.max(1)[1].unsqueeze(1).data, target.data)
            # print('acc:{}, {}'.format(nc1 / target.size(0), nc5 / target.size(0)))

    val_loss /= len(data_loader)
    iou, iou_classes = iou_eval.getIoU()

    return val_loss, iou, iou


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cross_entropy(input, target, label_smoothing=0.0, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets (long tensor)
         size_average: if false, sum is returned instead of mean
    Examples::
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    if label_smoothing <= 0.0:
        return F.cross_entropy(input, target)
    assert input.dim() == 2 and target.dim() == 1
    target_ = torch.unsqueeze(target, 1)
    one_hot = torch.zeros_like(input)
    one_hot.scatter_(1, target_, 1)
    one_hot = torch.clamp(one_hot, max=1.0-label_smoothing, min=label_smoothing/(one_hot.size(1) - 1.0))

    if size_average:
        return torch.mean(torch.sum(-one_hot * F.log_softmax(input, dim=1), dim=1))
    else:
        return torch.sum(torch.sum(-one_hot * F.log_softmax(input, dim=1), dim=1))


def joint_loss(model, data, target, teacher_model, distill, label_smoothing=0.0, mixup_alpha=None):
    if mixup_alpha is not None:
        data, targets_a, targets_b, lam = mixup_data(data, target, mixup_alpha)
        temp_criterion = lambda pred, y: cross_entropy(pred, y, label_smoothing=label_smoothing)
        criterion = lambda pred, y: mixup_criterion(temp_criterion, pred, y, targets_b, lam)
    else:
        criterion = lambda pred, y: cross_entropy(pred, y, label_smoothing=label_smoothing)

    output = model(data)
    if distill <= 0.0:
        return criterion(output, target)
    else:
        with torch.no_grad():
            teacher_output = teacher_model(data).data
        distill_loss = torch.mean((output - teacher_output) ** 2)
        if distill >= 1.0:
            return distill_loss
        else:
            class_loss = criterion(output, target)
            # print("distill loss={:.4e}, class loss={:.4e}".format(distill_loss, class_loss))
            return distill * distill_loss + (1.0 - distill) * class_loss


class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        enc = False
        weight = torch.ones(20)
        if (enc):
            weight[0] = 2.3653597831726
            weight[1] = 4.4237880706787
            weight[2] = 2.9691488742828
            weight[3] = 5.3442072868347
            weight[4] = 5.2983593940735
            weight[5] = 5.2275490760803
            weight[6] = 5.4394111633301
            weight[7] = 5.3659925460815
            weight[8] = 3.4170460700989
            weight[9] = 5.2414722442627
            weight[10] = 4.7376127243042
            weight[11] = 5.2286224365234
            weight[12] = 5.455126285553
            weight[13] = 4.3019247055054
            weight[14] = 5.4264230728149
            weight[15] = 5.4331531524658
            weight[16] = 5.433765411377
            weight[17] = 5.4631009101868
            weight[18] = 5.3947434425354
        else:
            weight[0] = 2.8149201869965
            weight[1] = 6.9850029945374
            weight[2] = 3.7890393733978
            weight[3] = 9.9428062438965
            weight[4] = 9.7702074050903
            weight[5] = 9.5110931396484
            weight[6] = 10.311357498169
            weight[7] = 10.026463508606
            weight[8] = 4.6323022842407
            weight[9] = 9.5608062744141
            weight[10] = 7.8698215484619
            weight[11] = 9.5168733596802
            weight[12] = 10.373730659485
            weight[13] = 6.6616044044495
            weight[14] = 10.260489463806
            weight[15] = 10.287888526917
            weight[16] = 10.289801597595
            weight[17] = 10.405355453491
            weight[18] = 10.138095855713

        weight[19] = 0
        # self.loss = torch.nn.NLLLoss2d(weight)
        self.loss = torch.nn.NLLLoss(weight=weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)

cityscapes_criterion = CrossEntropyLoss2d()


def joint_loss_cityscape(model, data, target, teacher_model, distill):
    output = model(data)
    if target.is_cuda:
        cityscapes_criterion.cuda()
    else:
        cityscapes_criterion.cpu()

    if distill <= 0.0:
        return cityscapes_criterion(output, target[:, 0])
    else:
        with torch.no_grad():
            teacher_output = teacher_model(data).data
        distill_loss = torch.mean((output - teacher_output) ** 2)
        if distill >= 1.0:
            return distill_loss
        else:
            class_loss = cityscapes_criterion(output, target[:, 0])
            # print("distill loss={:.4e}, class loss={:.4e}".format(distill_loss, class_loss))
            return distill * distill_loss + (1.0 - distill) * class_loss


def column_sparsity_common(model, out=None, verbose=False):
    if out is None:
        res = []
    else:
        res = out
    i = 0
    last_output_width = None
    for name, p in model.named_parameters():
        if name.endswith('weight') and p.dim() > 1:
            input_width = p.size(1)
            if last_output_width is not None and input_width != last_output_width:
                # the first fc layer after conv layers
                assert input_width > last_output_width and input_width % last_output_width == 0
                input_width = last_output_width
            p_t = p.data.transpose(0, 1).contiguous().view(input_width, -1)
            a = torch.sum(p_t ** 2, dim=1)
            if verbose:
                print("layer {:>20} channel-wise norm: min={:.4e}, mean={:.4e}, max={:.4e}".format(name, a.min(),
                                                                                                   a.mean(),
                                                                                                   a.max()))
            if out is None:
                res.append(a.nonzero().size(0))
            else:
                res[i] = float(a.nonzero().size(0))
            last_output_width = p.size(0)
            i += 1
    return res


def column_sparsity_resnet(model, out=None, verbose=False):
    if out is None:
        res = []
    else:
        res = out
    if isinstance(model, torch.nn.DataParallel):
        p_list = model.module.get_cp_weights()
    else:
        p_list = model.get_cp_weights()
    for i, p in enumerate(p_list):
        if p is None:
            # the input layer
            assert i == 0
            if out is None:
                res.append(3)
            else:
                res[i] = 3.0
        else:
            input_width = p.size(1)
            p_t = p.data.transpose(0, 1).contiguous().view(input_width, -1)
            a = torch.sum(p_t ** 2, dim=1)
            if verbose:
                print("layer {:>20} channel-wise norm: min={:.4e}, mean={:.4e}, max={:.4e}".format(i, a.min(),
                                                                                                   a.mean(),
                                                                                                   a.max()))
            if out is None:
                res.append(a.nonzero().size(0))
            else:
                res[i] = float(a.nonzero().size(0))
    return res


def column_sparsity_mbnet(model, out=None, verbose=False, zero_pre=False):
    if out is None:
        res = []
    else:
        res = out
    # the first layer is normal conv, the last layer is fc, grouped conv are in the middle
    nlayers = len(Mobilenet_width_ub) - 1
    W = []
    for name, p in model.named_parameters():
        if name.endswith('weight') and p.dim() > 1:
            W.append(p.data)
    z_idx = []
    w_tobezero = []
    for layer_idx in range(nlayers):
        i = layer_idx * 2 - 1
        if 0 < layer_idx < nlayers - 1:
            p1 = W[i]
            p2 = W[i + 1]
            assert p1.size(0) == p2.size(1) and p1.size(1) == 1
            p2_t = p2.data.transpose(0, 1).contiguous()
            p2_t = p2_t.view(p2.size(1), -1)
            a = torch.sum(p1.view(p1.size(0), -1) ** 2, dim=1) + torch.sum(p2_t ** 2, dim=1)
        elif layer_idx == 0:
            p = W[i + 1]
            assert p.dim() == 4 and i + 1 == 0
            p_t = p.data.transpose(0, 1).contiguous()
            p_t = p_t.view(p.size(1), -1)
            a = torch.sum(p_t ** 2, dim=1)
        else:
            p = W[i]
            assert p.dim() == 2 and i == len(W) - 1
            a = torch.sum(p ** 2, dim=0)

        if verbose:
            print("layer {} channel-wise norm: min={:.4e}, mean={:.4e}, max={:.4e}".format(layer_idx, a.min(), a.mean(),
                                                                                           a.max()))
        if out is None:
            res.append(a.nonzero().size(0))
        else:
            res[layer_idx] = float(a.nonzero().size(0))

        if zero_pre:
            z_idx.append(a == 0.0)
            if 0 < layer_idx < nlayers - 1:
                w_tobezero.append(p2)
            else:
                w_tobezero.append(p)

    if zero_pre:
        for i in range(len(z_idx)-1, 0, -1):
            if w_tobezero[i-1].dim() == 4:
                w_tobezero[i-1].data[z_idx[i], :, :, :] = 0.0
            else:
                assert False
                assert w_tobezero[i-1].dim() == 2
                w_tobezero[i-1].data[z_idx[i], :] = 0.0

    return res


def column_norm_mean(model):
    res = []
    i = 0
    last_output_width = None
    for name, p in model.named_parameters():
        if name.endswith('weight'):
            input_width = p.size(1)
            if last_output_width is not None and input_width != last_output_width:
                # the first fc layer after conv layers
                assert input_width > last_output_width and input_width % last_output_width == 0
                input_width = last_output_width
            p_t = p.data.transpose(0, 1).contiguous().view(input_width, -1)
            a = torch.sum(p_t ** 2, dim=1)
            res.append(a.mean())
            last_output_width = p.size(0)
            i += 1
    return res


def argmax(a):
    return max(range(len(a)), key=a.__getitem__)


def array1d_repr(t):
    res = ''
    for i in range(len(t)):
        res += '{:.3f}'.format(float(t[i]))
        if i < len(t) - 1:
            res += ', '

    return '[' + res + ']'


def model_grad_sqnorm(model):
    res = 0.0
    for p in model.parameters():
        if p.grad is not None:
            res += (p.grad.data ** 2).sum().item()
    return res


def filter_projection_resnet(model, layer_idx, num_filters):
    if isinstance(model, torch.nn.DataParallel):
        p_list = model.module.get_cp_weights()
    else:
        p_list = model.get_cp_weights()

    i = 0
    assert layer_idx > 0
    for p in p_list:
        if i == layer_idx:
            input_width = p.size(1)
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
                assert False
        i += 1


def is_cuda(x):
    if isinstance(x, torch.nn.Module):
        return next(x.parameters()).is_cuda
    elif isinstance(x, torch.Tensor):
        return x.is_cuda
    else:
        raise ValueError('unsupported data type')