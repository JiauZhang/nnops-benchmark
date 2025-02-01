import pyperf
import nnops.tensor, nnops.ops
import numpy as np

np_a = np.random.randn(8, 3, 512, 512)
np_b = np.random.randn(8, 3, 512, 512)
nps_a = nnops.tensor.from_numpy(np_a)
nps_b = nnops.tensor.from_numpy(np_b)
np_a_stride = np_a[2::3, ::2, ::3, 1::2]
np_b_stride = np_b[2::3, ::2, ::3, 1::2]
nps_a_stride = nps_a[2::3, ::2, ::3, 1::2]
nps_b_stride = nps_b[2::3, ::2, ::3, 1::2]

def np_add():
    _ = np_a + np_b

def nps_add():
    _ = nnops.ops.add(nps_a, nps_b)

def np_add_stride():
    _ = np_a_stride + np_b_stride

def nps_add_stride():
    _ = nnops.ops.add(nps_a_stride, nps_b_stride)

def np_mul():
    _ = np_a * np_b

def nps_mul():
    _ = nnops.ops.mul(nps_a, nps_b)

def np_mul_stride():
    _ = np_a_stride * np_b_stride

def nps_mul_stride():
    _ = nnops.ops.mul(nps_a_stride, nps_b_stride)

runner = pyperf.Runner()
runner.bench_func('numpy add', np_add)
runner.bench_func('nnops add', nps_add)
runner.bench_func('numpy add with stride', np_add_stride)
runner.bench_func('nnops add with stride', nps_add_stride)
runner.bench_func('numpy mul', np_mul)
runner.bench_func('nnops mul', nps_mul)
runner.bench_func('numpy mul with stride', np_mul_stride)
runner.bench_func('nnops mul with stride', nps_mul_stride)
