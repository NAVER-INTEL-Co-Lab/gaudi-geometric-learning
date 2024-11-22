# Build the shared library for METIS partition from C source

Install METIS
```shell
wget https://web.archive.org/web/20211119110155/http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
tar -xzf metis-5.1.0.tar.gz
cd metis-5.1.0
```

Edit the file `include/metis.h` and specify the width (32 or 64 bits) of the elementary data type used in METIS. This is controled by the `IDXTYPEWIDTH` constant.

For now, on a 32 bit architecture you can only specify a width of 32, whereas for a 64 bit architecture you can specify a width of either 32 or 64 bits.

```shell
make config shared=1 prefix=/usr/local cc=gcc
make
make insatll
```

Build the C source files

```shell
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export WITH_METIS=1
mkdir build
cd build
export CMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake ..
make
```

Load the shared library

```python
torch.ops.load_library("PATH/TO/METIS_PARITION/csrc/build/libtsmetis.so")
```

Then you can use

```python
torch.ops.torch_sparse.partition
```
