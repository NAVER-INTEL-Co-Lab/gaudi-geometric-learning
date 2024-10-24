# Handbook for Geometric Deep Learning with Intel Gaudi-v2 Devices

In this repository, we aim to provide instructions for geometric deep learning using Intel Gaudi-v2 devices. Specifically, we aim to provide the following contents:

- Efficient implementation of geometric deep learning functionalities and models that is compatible to Intel Gaudi-v2 devices.
- Detailed instructions and automatic tools for code adapatation to make existing code on geometric deep learning compatible to Intel Gaudi-v2 devices.
- Comprehensive benchmarks and tools for analyzing the performance of geometric deep learning models on Intel Gaudi-v2 devices.
- Rich documentation with concrete examples and tutorials.

## Table of Contents

- [Introduction](#introduction)
- [Preparation](#preparation)
- [Contents](#contents)
- [Tutorials](#tutorials)

## Introduction

Geometric deep learning is a rapidly growing field that focuses on extending deep learning techniques to non-Euclidean domains such as graphs.
This repository aims to bridge the gap by providing support for Intel Gaudi-v2 devices, addressing the challenge that existing libraries like PyG (PyTorch Geometric) and DGL (Deep Graph Library) only support CUDA GPUs.

## Preparation

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

See more general instructions [here](https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html).

### 2. Install PyG (PyTorch Geometric)

From PyG 2.3 onwards, you can install and use PyG without any external library required except for PyTorch. For this, simply run:

```shell
pip install torch_geometric
```

See more instructions for the installation of PyG [here](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html).

## Tutorials

### 1. PyG Official Tutorials

In this part, we adapt and modify [PyG official tutorials](https://pytorch-geometric.readthedocs.io/en/stable/get_started/colabs.html) to be compatible to Intel Gaudi-v2 devices.


## Contents

Below are the detailed contents of this library.

### Raw-PyTorch implementation of torch-sparse functionalities

- [x] `scatter_sum` / `scatter_add`
- [x] `scatter_mean`
- [x] `scatter_max` / `scatter_min`
- [x] `sactter_mul`
- [x] `scatter_log_softmax`
- [x] `scatter_logsumexp`
- [x] `scatter_std`
