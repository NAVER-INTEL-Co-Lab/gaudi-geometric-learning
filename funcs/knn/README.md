# Build the shared library for knn from C source

Build the C source files

```shell
cd csrc
mkdir build
cd build
export CMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake ..
make
```

Load the shared library

```python
torch.ops.load_library("PATH/TO/KNN/csrc/build/libknn_graph.so")
```

Then you can use

```python
torch.ops.torch_cluster.knn
```