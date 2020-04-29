import torch
import os
from torch.utils.cpp_extension import load
from noname.torch import utils
from torch.nn.parallel import DistributedDataParallel as DDP

with_cuda = torch.cuda.is_available()
cflags = ['-g', '-Wall', '-std=c++14', '`pkg-config libzmq --cflags`']
cflags += ['-DHAVE_CUDA'] if with_cuda else []
base_dir = os.path.dirname(os.path.abspath(__file__))
c_lib = load(name='noname.torch.c_lib', sources=[os.path.join(base_dir, 'c_lib.cpp')], verbose=True,
             extra_cflags=cflags, with_cuda=with_cuda, extra_ldflags=['-lpthread', '`pkg-config libzmq --libs`'])


def push_pull_async_inplace(tensor: torch.Tensor, average=True, name=None, version=0, priority=0):
    # torch.ones()
    # print('push_pull_async ', name)
    assert torch.float32 == tensor.dtype, f"unsupported dtype '{tensor.dtype}'"
    return c_lib.push_pull_async_inplace(
        tensor, average, name, version, priority)

# def push_pull(tensor, average=True, name=None, version=0, priority=0, compression=Compression.none):
#   return tensor


def synchronize(handle):
    return c_lib.synchronize(handle)


def declare(name):
    print('declaring', name)
    c_lib.declare(name)


def declare_done():
    c_lib.declare_done()


def size(*args, **kwargs):
    return c_lib.size()


def local_size(*args, **kwargs):
    return 1  # TODO


def rank(*args, **kwargs):
    return 0  # TODO


def local_rank(*args, **kwargs):
    return 0  # TODO


def init():  # TODO: server init
    c_lib.init()


def shutdown():  # TODO
    if size() > 1: c_lib.shutdown()


def step():
    c_lib.step()
