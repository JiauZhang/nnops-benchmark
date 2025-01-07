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

runner = pyperf.Runner()
runner.bench_func('np_add', np_add)
runner.bench_func('nps_add', nps_add)
runner.bench_func('np_add_stride', np_add_stride)
runner.bench_func('nps_add_stride', nps_add_stride)
