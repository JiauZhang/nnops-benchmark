import nnops.tensor, nnops.ops
import numpy as np
import torch

np_a = np.random.randn(8, 3, 512, 512)
np_b = np.random.randn(8, 3, 512, 512)
nps_a = nnops.tensor.from_numpy(np_a)
nps_b = nnops.tensor.from_numpy(np_b)
th_a = torch.from_numpy(np_a)
th_b = torch.from_numpy(np_b)
np_a_stride = np_a[2::3, ::2, ::3, 1::2]
np_b_stride = np_b[2::3, ::2, ::3, 1::2]
nps_a_stride = nps_a[2::3, ::2, ::3, 1::2]
nps_b_stride = nps_b[2::3, ::2, ::3, 1::2]
th_a_stride = th_a[2::3, ::2, ::3, 1::2]
th_b_stride = th_b[2::3, ::2, ::3, 1::2]

def np_add():
    _ = np_a + np_b

def th_add():
    _ = th_a + th_b

def nps_add():
    _ = nnops.ops.add(nps_a, nps_b)

def np_add_stride():
    _ = np_a_stride + np_b_stride

def th_add_stride():
    _ = th_a_stride + th_b_stride

def nps_add_stride():
    _ = nnops.ops.add(nps_a_stride, nps_b_stride)

def np_mul():
    _ = np_a * np_b

def th_mul():
    _ = th_a * th_b

def nps_mul():
    _ = nnops.ops.mul(nps_a, nps_b)

def np_mul_stride():
    _ = np_a_stride * np_b_stride

def th_mul_stride():
    _ = th_a_stride * th_b_stride

def nps_mul_stride():
    _ = nnops.ops.mul(nps_a_stride, nps_b_stride)

bench_add = 'add'
bench_add_s = 'add with stride'
bench_mul = 'mul'
bench_mul_s = 'mul with stride'

bench_func = {
    'numpy': [
        (bench_add, np_add),
        (bench_add_s, np_add_stride),
        (bench_mul, np_mul),
        (bench_mul_s, np_mul_stride),
    ],
    'torch': [
        (bench_add, th_add),
        (bench_add_s, th_add_stride),
        (bench_mul, th_mul),
        (bench_mul_s, th_mul_stride),
    ],
    'nnops': [
        (bench_add, nps_add),
        (bench_add_s, nps_add_stride),
        (bench_mul, nps_mul),
        (bench_mul_s, nps_mul_stride),
    ],
}
