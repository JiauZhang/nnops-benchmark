import pyperf
import nnops.tensor, nnops.ops
import numpy as np
import torch

np_a = np.random.randn(8, 3, 512, 512).astype(np.float32)
np_b = np.random.randn(8, 3, 512, 512).astype(np.float32)
nps_a = nnops.tensor.from_numpy(np_a)
nps_b = nnops.tensor.from_numpy(np_b)
th_a = torch.from_numpy(np_a)
th_b = torch.from_numpy(np_b)
np_a_stride = np_a[2::3, ::2, ::3, 1::2]
np_b_stride = np_b[2::3, ::2, 1::2, 1::2]
nps_a_stride = nps_a[2::3, ::2, ::3, 1::2]
nps_b_stride = nps_b[2::3, ::2, 1::2, 1::2]
th_a_stride = th_a[2::3, ::2, ::3, 1::2]
th_b_stride = th_b[2::3, ::2, 1::2, 1::2]

def np_matmul():
    _ = np_a @ np_b

def th_matmul():
    _ = th_a @ th_b

def nps_matmul():
    _ = nnops.ops.matmul(nps_a, nps_b)

def np_matmul_stride():
    _ = np_a_stride @ np_b_stride

def th_matmul_stride():
    _ = th_a_stride @ th_b_stride

def nps_matmul_stride():
    _ = nnops.ops.matmul(nps_a_stride, nps_b_stride)

runner = pyperf.Runner()
runner.bench_func('numpy matmul', np_matmul)
runner.bench_func('torch matmul', th_matmul)
runner.bench_func('nnops matmul', nps_matmul)
runner.bench_func('numpy matmul with stride', np_matmul_stride)
runner.bench_func('torch matmul with stride', th_matmul_stride)
runner.bench_func('nnops matmul with stride', nps_matmul_stride)
