import torch
import os
from torch.utils.cpp_extension import load
from noname.nonametorch import utils
from torch.nn.parallel import DistributedDataParallel as DDP

with_cuda = torch.cuda.is_available()
cflags = ['-DHAVE_CUDA', '-O2', '-g', '-Wall', '-std=c++17', '`pkg-config libzmq --cflags`']
# cflags += [ if with_cuda else []

if 'ND' in os.environ: cflags += [f"-DNDEBUG"]
# if 'DMLC_RTT' in os.environ: cflags += [f"-DDMLC_RTT={os.environ['DMLC_RTT']}"]
# if 'DMLC_BUC_SZ' in os.environ: cflags += [f"-DDMLC_BUC_SZ={os.environ['DMLC_BUC_SZ']}"]
if 'MYLOG_LEVEL' in os.environ: cflags += [f"-DMYLOG_LEVEL={os.environ['MYLOG_LEVEL']}"]
if 'LOSS_RATE' in os.environ: cflags += [f"-DLOSS_RATE={100-int(os.environ['LOSS_RATE'])}"]
# if 'SEND_RATE' in os.environ: cflags += [f"-DSEND_RATE={os.environ['SEND_RATE']}"] # send rate
# if 'PP_METHOD' in os.environ:
#     assert os.environ['PP_METHOD'] in ['0', '1']
#     cflags += [f"-DPP_METHOD={os.environ['PP_METHOD']}"]
base_dir = os.path.dirname(os.path.abspath(__file__))
c_lib = load(name='noname.nonametorch.c_lib', sources=[os.path.join(base_dir, 'c_lib.cpp')], verbose=True,
             extra_cflags=cflags, with_cuda=with_cuda, extra_ldflags=['-lpthread', '`pkg-config libzmq --libs`'],
             build_directory=".")


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


def declare(name, idx, size):
    print('declaring', name, 'with idx', idx, 'size', size)
    c_lib.declare(name, idx, size)


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
