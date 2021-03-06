# ref: https://github.com/wbaek/torchskeleton/releases/tag/v0.2.1_dawnbench_cifar10_release
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed

class Mul(torch.nn.Module):
  def __init__(self, weight):
    super(Mul, self).__init__()
    self.weight = weight

  def forward(self, x):
    return x * self.weight


class Flatten(torch.nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1, bn=True, activation=True):
  op = [
      torch.nn.Conv2d(channels_in, channels_out,
              kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
  ]
  if bn:
    op.append(torch.nn.BatchNorm2d(channels_out))
  if activation:
    op.append(torch.nn.ReLU(inplace=True))
  return torch.nn.Sequential(*op)


class Residual(torch.nn.Module):
  def __init__(self, module):
    super(Residual, self).__init__()
    self.module = module

  def forward(self, x):
    return x + self.module(x)


def get_net_cls(aug, bn):
  print('aug: ', aug, 'bn: ', bn)
  class Resnet9(nn.Module):
    transform_train = transforms.Compose(
      ([transforms.RandomCrop(30, padding=2), transforms.RandomHorizontalFlip()] if aug else [])
      +
      [transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
      transforms.CenterCrop((30, 30)),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])

    def __init__(self):
      super(Resnet9, self).__init__()
      num_class = 10
      self.module = torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        # torch.nn.MaxPool2d(2),

        Residual(torch.nn.Sequential(
          conv_bn(128, 128),
          conv_bn(128, 128),
        )),

        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),

        Residual(torch.nn.Sequential(
          conv_bn(256, 256),
          conv_bn(256, 256),
        )),

        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),

        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, num_class, bias=False),
        Mul(0.2)
      )

    def init(self):
      for module in self.module.modules():
          if isinstance(module, torch.nn.BatchNorm2d):
              if hasattr(module, 'weight') and module.weight is not None:
                  module.weight.data.fill_(1.0)
              module.eps = 0.00001
              module.momentum = 0.1
          else:
              if args.half:
                  module.half()
          if isinstance(module, torch.nn.Conv2d) and hasattr(module, 'weight'):
              # torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))  # original
              torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='linear')
              # torch.nn.init.xavier_uniform_(module.weight, gain=torch.nn.init.calculate_gain('linear'))
          if isinstance(module, torch.nn.Linear) and hasattr(module, 'weight'):
              # torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))  # original
              torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='linear')
              # torch.nn.init.xavier_uniform_(module.weight, gain=1.)

    def forward(self, x):
      return self.module(x)
  return Resnet9

class LRScheduler:
  def __init__(self, optimizer, sched_fn):
    self.sched_fn = sched_fn
    self.optimizer = optimizer
    self.iters = 0
  def update(self, iters=None):
    if iters is not None: self.iters = iters
    lr = self.sched_fn(self.iters)
    for param_group in self.optimizer.param_groups:
      param_group['lr'] = lr
    self.iters += 1
    return lr
class myopt(optim.Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, prob = 0.1):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(myopt, self).__init__(params, defaults)
        self.prob_of_loss = prob

    def __setstate__(self, state):
        super(myopt, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
    # the gradient drop with the probility of 0.3
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
#                 tmp = p.grad.data * 100
#                 d_p = tmp.int().float() / 100
#                 d_p = p.grad.data.int().float()
                tmp = torch.rand(p.grad.data.size())
                tmp = (tmp > self.m).float()
                tmp = tmp.to(torch.device("cuda:0"))
                d_p = p.grad.data * tmp
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                
                p.data.add_(-group['lr'], d_p)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--prob', type=float, default=0.1)
parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                    help='input batch size for training (default: 500)')
parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
                    help='input batch size for testing (default: 500)')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train (default: 25)')
parser.add_argument('--lr', type=float, default=0.025, metavar='LR',
                    help='learning rate (default: 0.025)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.set_defaults(half=True)
parser.add_argument('--nohalf', action='store_false', dest='half',
                    help='not use fp16 in training')
parser.add_argument('--scale', action='store_true', default=False,
                    help='scale the loss by batch_size')
parser.add_argument('--fp16-pushpull', action='store_true', default=False,
                    help='use fp16 compression during pushpull')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
_scale = args.batch_size if args.scale else 1
# BytePS: initialize library.
torch.manual_seed(args.seed)

if args.cuda:
    # BytePS: pin GPU to local rank.
    # torch.cuda.set_device(bps.local_rank())
    torch.cuda.manual_seed(args.seed)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False

model = get_net_cls(True, True)()
if args.cuda:
    # Move model to GPU.
    model.cuda()
model.init()

if not os.path.exists('cifar10'): os.mkdir('cifar10')
kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
train_dataset = \
    datasets.CIFAR10('cifar10/data-%d' % 0, train=True, download=True,
                   transform=model.transform_train)
# BytePS: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=1, rank=0)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
    drop_last=True, **kwargs)

test_dataset = \
    datasets.CIFAR10('cifar10/data-%d' % 0, train=False, transform=model.transform_test)
# BytePS: use DistributedSampler to partition the test data.
test_sampler = None
#  torch.utils.data.distributed.DistributedSampler(
#     test_dataset, num_replicas=1, rank=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                          sampler=test_sampler, **kwargs)


# BytePS: scale learning rate by the number of GPUs.
optimizer = myopt(model.parameters(), lr=args.lr * 1,
                      momentum=args.momentum, weight_decay=5e-4*_scale, prob=args.prob)

# BytePS: (optional) compression algorithm.
# compression = bps.Compression.fp16 if args.fp16_pushpull else bps.Compression.none

# BytePS: wrap optimizer with DistributedOptimizer.
# optimizer = bps.DistributedOptimizer(optimizer,
#                                      named_parameters=model.named_parameters(),
#                                      compression=compression)

def get_change_scale(scheduler, init_scale=1.0):
    def schedule(e, scale=None, **kwargs):
        lr = scheduler(e, **kwargs)
        return lr * (scale if scale is not None else init_scale)
    return schedule

def get_piecewise(knots, vals):
    def schedule(e, **kwargs):
        return np.interp([e], knots, vals)[0]
    return schedule
lr_fn = get_change_scale(
    get_piecewise([0, 400, 2500], [0.025, 0.4, 0.001]),
      1.0 / _scale
)
lr_scheduler = LRScheduler(optimizer, lr_fn)


# BytePS: broadcast parameters.
# bps.broadcast_parameters(model.state_dict(), root_rank=0)
# bps.broadcast_optimizer_state(optimizer, root_rank=0)

def train(epoch):
    model.train()
    # BytePS: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        lr = lr_scheduler.update() * _scale
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        if args.half:
            data = data.half()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target, reduction='sum' if _scale != 1 else 'mean')
        loss.backward()
        optimizer.step()
       # if batch_idx % args.log_interval == 0:
            # BytePS: use train_sampler to determine the number of examples in
            # this worker's partition.
  #          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlr: {}'.format(
   #             epoch, batch_idx * len(data), len(train_sampler),
    #            100. * batch_idx / len(train_loader), loss.item() / _scale, lr))


# def metric_average(val, name):
#     tensor = torch.tensor(val)
#     avg_tensor = bps.push_pull(tensor, name=name)
#     return avg_tensor.item()


def test():
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        if args.half:
            data = data.half()
        output = model(data)
        # sum up batch loss
        test_loss += F.cross_entropy(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

    # BytePS: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_loader.dataset)
    test_accuracy /= len(test_loader.dataset)

    # BytePS: average metric values across workers.
    # test_loss = metric_average(test_loss, 'avg_loss')
    # test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

    # BytePS: print output only on first rank.
    if 0 == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))


import time
st = time.time()
for epoch in range(1, 26):
    train(epoch)
    test()
#    print("Epoch time: %ss"%((time.time() - st) / epoch))
# pri41nt('total train time=%ss'%(time.time() - st))
