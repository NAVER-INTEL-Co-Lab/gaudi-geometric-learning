# PyG Tutorials with Intel Gaudi-v2 Devices

The tutorials here are adapted from [PyG official tutorials](https://pytorch-geometric.readthedocs.io/en/stable/get_started/colabs.html).
The original source code can be found [here](https://github.com/AntonioLonga/PytorchGeometricTutorial).
You can also find their video tutorials on [Youtube](https://www.youtube.com/user/94longa2112/featured) and at their official website [here](https://antoniolonga.github.io/Pytorch_geometric_tutorials/index.html).

## Contents

### ✅ Tutorial 1: What is Geometric Deep Learning?

Adaptation completed.
See the [subfolder](Tutorial1/).
See the official video [here](https://youtu.be/JtDgmmQ60x8).

### ✅ Tutorial 2: PyTorch basics

Adaptation completed.
See the [subfolder](Tutorial2/).
See the official video [here](https://youtu.be/UHrhp2l_knU).

### ✅ Tutorial 3: Graph Attention Network GAT

Adaptation completed.
See the [subfolder](Tutorial3/).
See the official video [here](https://youtu.be/CwsPoa7z2c8).

### ✅ Tutorial 4: Convolutional Layers - Spectral methods

Adaptation completed.
See the [subfolder](Tutorial4/).
See the official video [here](https://youtu.be/Ghw-fp_2HFM).

### ❌ Tutorial 5: Aggregation Functions in GNNs

Currently, we encounter `RuntimeError: Input sizes must be equal` and are debugging.
See the error messages [here](Tutorial5/error.pdf).
See the [subfolder](Tutorial4/).
See the official video [here](https://youtu.be/tGXovxQ7hKU).

### ✅ Tutorial 6: Graph Autoencoders and Variational Graph Autoencoders

Adaptation completed.
See the [subfolder](Tutorial6/).
See the official video [here](https://youtu.be/qA6U4nIK62E).

### ✅ Tutorial 7: Adversarially regularized GAE and VGAE

Adaptation completed.
See the [subfolder](Tutorial7/).
See the official video [here](https://youtu.be/hZkLu2OaHD0).

### ✅ Tutorial 8: Graph Generation

PDF only.
See the [subfolder](Tutorial8/).
See the official video [here](https://youtu.be/embpBq1gHAE).

### ✅ Tutorial 9: Recurrent Graph Neural Networks Open In Colab

Adaptation completed.
See the [subfolder](Tutorial9/).
See the official video [here](https://youtu.be/v7TQ2DUoaBY).

### ✅ Tutorial 10: DeepWalk and Node2Vec (Theory)

PDF only.
See the [subfolder](Tutorial10/).
See the official video [here](https://youtu.be/QZQBnl1QbCQ).

### ❌ Tutorial 11: DeepWalk and Node2Vec (Practice)

Currently, we cannot run `Node2Vec` which depends on `torch_cluster`, while `torch_cluster` only supports CUDA GPUs.
See the error messages [here](Tutorial11/error.pdf).
See the [subfolder](Tutorial11/).
See the official video [here](https://youtu.be/5YOcpI3dB7I).

### ❌ Tutorial 12: Edge analysis Open In Colab

For "GAE for link prediction", adaptation completed.
For "Node2Vec for label prediction", currently, we cannot run `Node2Vec` which depends on `torch_cluster`, while `torch_cluster` only supports CUDA GPUs.
The problem is the same as in Tutorial 11.
See the error messages [here](Tutorial11/error.pdf).
See the [subfolder](Tutorial12/).
See the official video [here](https://youtu.be/m1G7oS9hmwE).

### ❌ Tutorial 13: Metapath2vec

Currently, we encounter `NotImplementedError: Could not run 'aten::_sparse_coo_tensor_with_dims_and_tensors' with arguments from the 'SparseHPU' backend.` and are debugging.
However, we are able to adapt the code without sparse computation.
See the error messages [here](Tutorial13/error.pdf).
See the [subfolder](Tutorial13/).
See the official video [here](https://youtu.be/GtPoGehuKYY).

### ❌ Tutorial 14: Data handling in Pyg (part 1)

Currently, we cannot run `ClusterData` and `NeighborSampler`:

- `ClusterData` requires either `pyg-lib` or `torch-sparse`.
- `NeighborSampler` requires `SparseTensor`, which requires `torch-sparse`.
See the error messages [here](Tutorial14/error.pdf).
See the [subfolder](Tutorial14/).
See the official video [here](https://youtu.be/Vz5bT8Xw6Dc).

### ✅ Tutorial 15: Data handling in Pyg (part 2)

Adaptation completed.
See the [subfolder](Tutorial15/).
See the official video [here](https://youtu.be/Q5T-JdyVCfs).

### ❌ Tutorial 16: Graph pooling: DIFFPOOL

The original tutorial has bugs.
See the official video [here](https://youtu.be/Uqc3O3-oXxM).

### Special guest talk 1: Matthias Fey

No code.
See the official video [here](https://youtu.be/MA6VH7Vwtb4).

### Special guest talk 2: Sergei Ivanov

No code.
See the official video [here](https://youtu.be/hX297pr1RHE).
