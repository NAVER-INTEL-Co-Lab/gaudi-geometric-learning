# Handbook for Geometric Deep Learning with Intel Gaudi-v2 Devices

In this repository, we aim to provide instructions for geometric deep learning using Intel Gaudi-v2 devices. Specifically, we aim to provide the following contents:

- Efficient implementation of geometric deep learning functionalities and models that is compatible to Intel Gaudi-v2 devices.
- Detailed instructions and automatic tools for code adaptation to make existing code on geometric deep learning compatible to Intel Gaudi-v2 devices.
- Comprehensive benchmarks and tools for analyzing the performance of geometric deep learning models on Intel Gaudi-v2 devices.
- Rich documentation with concrete examples and tutorials.

## Table of Contents

- [Introduction](#introduction)
- [Preparation](#preparation)
- [Tutorials and Examples](#tutorials-and-examples)
- [Benchmark Results](#benchmark-results)
- [Implemented Functionalities](#implemented-functionalities)  
- [Performance Optimization](#performance-optimization)

## Introduction

Geometric deep learning is a rapidly growing field that focuses on extending deep learning techniques to non-Euclidean domains such as graphs.
This repository aims to bridge the gap by providing support for Intel Gaudi-v2 devices, addressing the challenge that existing libraries like PyG (PyTorch Geometric) and DGL (Deep Graph Library) only support CUDA GPUs.

## Preparation

In this section, we show what you should prepare for running code with PyG on Intel Gaudi-v2 devices.

### 1. Install PyTorch on Gaudi-v2

First, install PyTorch on Gaudi-v2. Make sure to install the PyTorch packages provided by Intel Gaudi. To set up the PyTorch environment, refer to the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html#gaudi-installation-guide). The supported PyTorch versions are listed in the [Support Matrix](https://docs.habana.ai/en/latest/Support_Matrix/Support_Matrix.html#support-matrix). The recommended way is to use a docker container with PyTorch installed.
See the instructions for installation using containers [here](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html).
An example for the installation of PyTorch 2.4.0 on Ubuntu 22.04 is

```shell
export OS="ubuntu22.04";
export PTV="2.4.0";
docker pull vault.habana.ai/gaudi-docker/1.18.0/${OS}/habanalabs/pytorch-installer-${PTV}:latest;
docker run --name torch${PTV} -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host -v /home/irteamsu/data:/data vault.habana.ai/gaudi-docker/1.18.0/$OS/habanalabs/pytorch-installer-${PTV}:latest;
```

Check the available docker list [here](https://vault.habana.ai/ui/native/gaudi-docker/).
See more general instructions [here](https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html).

### 2. Install PyG (PyTorch Geometric)

From PyG 2.3 onwards, you can install and use PyG without any external library required except for PyTorch. For this, simply run:

```shell
pip install torch_geometric
```

See more instructions for the installation of PyG [here](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html).

### 3. Basic Steps to Adapt Code

There are several basic steps to adapt code to be compatible to Intel Gaudi-v2 devices.

The first step is to import the Intel Gaudi PyTorch framework:

```Python
import habana_frameworks.torch.core as htcore
```

The second step is to set the device as the Gaudi device:

```Python
device = torch.device("hpu")
```

The third step is to wrap the model in `torch.compile` function and set the backend to `hpu_backend`, after `model.train()`

```Python
model.train()
model = torch.compile(model,backend="hpu_backend")
```

Finally, run the code in the Eager mode. To do this, you can either set the environment variable when running the code, e.g.,

```shell
PT_HPU_LAZY_MODE=0 python main.py
```

Or set the environment variable in the beginning of the code:

```Python
import os
# Use the eager mode
os.environ["PT_HPU_LAZY_MODE"] = "0"
```

See more instructions [here](https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html).
See also the introduction of an automatic migration toolkit [here](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Model_Porting/GPU_Migration_Toolkit/GPU_Migration_Toolkit.html).

## Tutorials and Examples

In this section, we provide tutorials and examples for geometric deep learning with Intel Gaudi-v2 devices.

### 1. PyG Official Tutorials

In this part, we adapt and modify [PyG official tutorials](https://github.com/AntonioLonga/PytorchGeometricTutorial) to be compatible to Intel Gaudi-v2 devices.
See the [subfolder](tutorials_and_examples/pyg_tutorials/) for more details.

### 2. PyG Official Example

In this part, we adapt and modify [PyG official examples](https://pytorch-geometric.readthedocs.io/en/stable/get_started/colabs.html) to be compatible to Intel Gaudi-v2 devices.
See the [subfolder](tutorials_and_examples/pyg_examples/) for more details.

### 3. Stanford CS224W Tutorials

In this part, we adapt and modify [Stanford CS224W Tutorials](https://medium.com/stanford-cs224w) to be compatible to Intel Gaudi-v2 devices.
See the [subfolder](tutorials_and_examples/stanford_cs224w/) for more details.

## Benchmark Results

In this section, we provide benchmark results for geometric deep learning models on Intel Gaudi-v2 devices.

TODO: This section will showcase the benchmark results comparing Intel Gaudi-v2 devices with other hardware platforms (e.g., NVIDIA GPUs) for various geometric deep learning models and tasks. The benchmarks will include:

- Training time comparisons
- Memory usage analysis
- Cost-performance ratio evaluation
- Scaling efficiency with different batch sizes
- Model inference latency measurements

## Implemented Functionalities

In this section, we provide implementations of geometric deep learning functionalities that are compatible to Intel Gaudi-v2 devices.

### 1. Raw-PyTorch Implementation of Torch-Scatter Functionalities

We implement main functionalities in the Torch Scatter library, which only supports CUDA GPUs.
We implement them using raw Pytorch so that the implementation can be compatible to Intel Gaudi-v2 devices.

- [x] `scatter_sum` / `scatter_add`
- [x] `scatter_mean`
- [x] `scatter_max` / `scatter_min`
- [x] `sactter_mul`
- [x] `scatter_log_softmax`
- [x] `scatter_logsumexp`
- [x] `scatter_std`

## Performance Optimization

[Official optimization instructions](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_Getting_Started.html)
