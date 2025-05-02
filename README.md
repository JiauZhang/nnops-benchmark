```shell
# benchmark
python3 benchmark.py --backend nnops --output nnops.json
# compare
pyperf compare_to nnops-x.json nnops-y.json
```