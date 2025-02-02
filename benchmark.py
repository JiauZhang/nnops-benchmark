import argparse, pyperf, importlib, sys
import torch, numpy, nnops

# python3 benchmark.py --target matmul --backend nnops

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default='all')
parser.add_argument('--backend', type=str, default='all')

def add_cmdline_args(cmd, args):
    cmd += ['--target', args.target, '--backend', args.backend]

runner = pyperf.Runner(_argparser=parser, add_cmdline_args=add_cmdline_args)
args = runner.parse_args()
targets = [
    'binary_ops', 'matmul',
]
backends = [
    'numpy', 'torch', 'nnops'
]

if args.target != 'all':
    target = args.target.split()
else:
    target = targets
if args.backend != 'all':
    backend = args.backend.split()
else:
    backend = backends

for tgt in target:
    if tgt not in targets:
        raise RuntimeError(f'target {tgt} not exists!')
    tgt = importlib.import_module(tgt)
    bench_func = tgt.bench_func

    bk_funcs = []
    for bk in backend:
        if bk not in backends:
            raise RuntimeError(f'backend {bk} not exists!')
        bk_funcs.append([])
        for name, func in bench_func[bk]:
            bk_funcs[-1].append((f'{bk} {name}', func))

    for named_funcs in zip(*bk_funcs):
        for name, func in named_funcs:
            runner.bench_func(name, func)
