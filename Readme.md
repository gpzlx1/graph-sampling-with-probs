# Graph-sampling-with-probs
```shell
.
├── CMakeLists.txt
├── Readme.md
├── src
│   ├── cuda_ops.cuh
│   ├── rowwise_sampling.cu
│   ├── rowwise_sampling_alias.cu
│   ├── rowwise_sampling_ares.cu
│   ├── rowwise_sampling_cdf.cu
│   ├── test.cu
│   └── warpselect
│       ├── MergeNetworkWarp.cuh
│       ├── Pair.cuh
│       ├── Reductions.cuh
│       ├── Utils.cuh
│       └── WarpSelect.cuh
└── test
    ├── bench.py
    ├── load_graph.py
    ├── test.py
    ├── test_rowwise_sampling_alias.py      # alias sampling test and bench
    ├── test_rowwise_sampling_ares.py       # ares sampling test and bench
    ├── test_rowwise_sampling_cdf.py        # cdf sampling test and bench
    └── test_rowwise_sampling_uniform.py    # uniform sampling test and bench
```
## Compilation
Requirement:
* pytorch >= 1.10
* DGL >= 0.8

How to compile:
```shell
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
$ cd -
```

## Run demo
```shell
# run alias sampling w/ bias (only replace=True)
$ python test/test_rowwise_sampling_alias.py

# run cdf sampling w/ bias (only replace=True)
$ python test/test_rowwise_sampling_cdf.py

# run A-Res sampling w/ bias (only replace=False)
$ python test/test_rowwise_sampling_ares.py

# run uniform sampling w/o bias
$ python test/test_rowwise_sampling_uniform.py
```
