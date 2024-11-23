# Build the shared library for "neighbor sample" from C source

Build the C source files

```shell
cd csrc
mkdir build
cd build
export CMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake ..
make
```

**Note:** The files in `cpu/parallel_hashmap` are from [here](https://github.com/greg7mdp/parallel-hashmap).

Load the shared library

```python
torch.ops.load_library("PATH/TO/NEIGHBOR_SAMPLE/csrc/build/libneighbor_sample.so")
```

Then you can use

```python
torch.ops.torch_sparse.neighbor_sample
torch.ops.torch_sparse.hetero_neighbor_sample
torch.ops.torch_sparse.hetero_temporal_neighbor_sample
```