# Build the shared library for random walk from C source

Build the C source files

```shell
mkdir build
cd build
export CMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake ..
make
```

Load the shared library

```python
torch.ops.load_library("PATH/TO/RANDOM_WALK/csrc/build/librandom_walk.so")
```

Then you can use

```python
torch.ops.torch_cluster.random_walk
```
