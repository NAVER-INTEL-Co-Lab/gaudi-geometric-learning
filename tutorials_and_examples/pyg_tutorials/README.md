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

See the [subfolder](Tutorial5/).

See the official video [here](https://youtu.be/tGXovxQ7hKU).

#### Some debugging information

The code works well on CPU. See the code on CPU [here](Tutorial5/Tutorial5_cpu.ipynb).

After removing `model = torch.compile(model, backend="hpu_backend")`, we encounter

```plaintext
[Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in Lowering thread...
synStatus=1 [Invalid argument] Node reshape  failed.
```

See the error messages [here](Tutorial5/error_no_compile.pdf).

See the related question on Intel Gaudi forum [here](https://forum.habana.ai/t/runtimeerror-input-sizes-must-be-equal-when-doing-loss-backward-during-the-training-of-a-gnn/1331).

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

See the related issue on Pytorch Cluster GitHub repo [here](https://github.com/rusty1s/pytorch_cluster/issues/230).

See the related discussion on PyG GitHub repo [here](https://github.com/pyg-team/pytorch_geometric/discussions/9760).

See the related question on Intel Gaudi forum [here]().

### ❌ Tutorial 12: Edge analysis Open In Colab

For "GAE for link prediction", adaptation completed.
For "Node2Vec for label prediction", currently, we cannot run `Node2Vec` which depends on `torch_cluster`, while `torch_cluster` only supports CUDA GPUs.
The problem is the same as in [Tutorial 11](#-tutorial-11-deepwalk-and-node2vec-practice).
See the error messages [here](Tutorial11/error.pdf).

See the [subfolder](Tutorial12/).

See the official video [here](https://youtu.be/m1G7oS9hmwE).

See the related question on Intel Gaudi forum [here](https://forum.habana.ai/t/notimplementederror-could-not-run-aten-sparse-coo-tensor-with-dims-and-tensors-with-arguments-from-the-sparsehpu-backend/1330).

### ❌ Tutorial 13: Metapath2vec

Currently, we encounter `NotImplementedError: Could not run 'aten::_sparse_coo_tensor_with_dims_and_tensors' with arguments from the 'SparseHPU' backend.` and are debugging.
Seemingly, sparse operations have not been supported by Guadi yet.
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

Currently, we encounter

```plaintext
RuntimeError: [Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in Lowering thread...
[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT Run returned exception....
Graph compile failed. synStatus=synStatus 26 [Generic failure]. 
[Rank:0] Habana exception raised from compile at graph.cpp:599
[Rank:0] Habana exception raised from LaunchRecipe at graph_exec.cpp:558
```

and are debugging.
See the error messages [here](Tutorial16/error.pdf).

See the [subfolder](Tutorial16/).

See the official video [here](https://youtu.be/Uqc3O3-oXxM).

### Special guest talk 1: Matthias Fey

No code.
See the official video [here](https://youtu.be/MA6VH7Vwtb4).

### Special guest talk 2: Sergei Ivanov

No code.
See the official video [here](https://youtu.be/hX297pr1RHE).
