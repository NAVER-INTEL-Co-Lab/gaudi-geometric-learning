# PyG Examples with Intel Gaudi-v2 Devices

The examples here are adapted from [PyG official examples](https://pytorch-geometric.readthedocs.io/en/stable/get_started/colabs.html).

## Contents

### ✅ Example 1: Introduction

Adaptation completed.

See the [subfolder](Example1/).

### ✅ Example 2: Node Classification

Adaptation completed.

See the [subfolder](Example2/).

### 🟡 Example 3: Graph Classification

Currently, we encounter

```plaintext
RuntimeError: [Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in Lowering thread...
[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT Run returned exception....
synNodeCreateWithId failed for node: concat with synStatus 1 [Invalid argument]. .
[Rank:0] Habana exception raised from add_node at graph.cpp:507
[Rank:0] Habana exception raised from LaunchRecipe at graph_exec.cpp:558
```

and are debugging. See the error messages [here](Example3/error.pdf).

**Workaround:** We are able to adapt the code by removing `model = torch.compile(model, backend="hpu_backend")`.
See such adapted code [here](Example3/3_Graph_Classification_no_compile.ipynb).
See also the debugging information [below](#example-3-debugging-information).

See the [subfolder](Example3/).

See the related question on Intel Gaudi forum [here](https://forum.habana.ai/t/gcnconv-fails-with-normalization/1336).

#### Example 3: Debugging Information

Seemingly, the problem is caused by the function `gcn_norm` since it works well when we set `normalize = False` for `GCNConv`.
See the code without normalization [here](Example3/3_Graph_Classification_no_normalize.ipynb).
After removing `model = torch.compile(model, backend="hpu_backend")`, it works even with `normalize = True`.

### 🟡 Example 4: Scaling GNNs

Currently, we cannot run `ClusterData`, which requires either `pyg-lib` or `torch-sparse`, while both `pyg-lib` and `torch-sparse` only support CUDA GPUs.
See the error messages [here](Example4/error.pdf).

**Workaround (ClusterData):** We are able to adapt the code by building the realted functions (specifically, METIS parition) in `torch_sparse` from C source files. See more details [here](../../funcs/metis_partition/).

**Workaround (GCNConv):** We are able to adapt the code by removing `model = torch.compile(model, backend="hpu_backend")`. See more details above in [Example 3](#-example-3-graph-classification).

See the [subfolder](Example4/).

See the related issue on Pytorch Cluster GitHub repo [here](https://github.com/rusty1s/pytorch_cluster/issues/230).

See the related discussion on PyG GitHub repo [here](https://github.com/pyg-team/pytorch_geometric/discussions/9760).

See the related question on Intel Gaudi forum [here](https://forum.habana.ai/t/functionalities-that-require-pyg-lib-torch-sparse-torch-cluster/1363).

### ❌ Example 5: Point Cloud Classification

Currently, we cannot run `knn_graph`, which requires `torch-cluster`, while `torch-cluster` only supports CUDA GPUs.
See the error messages [here](Example5/error.pdf).
See the [subfolder](Example5/).

See the related issue on Pytorch Cluster GitHub repo [here](https://github.com/rusty1s/pytorch_cluster/issues/230).

See the related discussion on PyG GitHub repo [here](https://github.com/pyg-team/pytorch_geometric/discussions/9760).

See the related question on Intel Gaudi forum [here](https://forum.habana.ai/t/functionalities-that-require-pyg-lib-torch-sparse-torch-cluster/1363).

#### Example 5: Debugging Information

We manage to build the required `knn` operation from C source files.
However, we encounter new errors

```plaintext
RuntimeError: [Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in Lowering thread...
[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT Run returned exception....
synNodeCreateWithId failed for node: concat with synStatus 1 [Invalid argument]. .
[Rank:0] Habana exception raised from add_node at graph.cpp:507
[Rank:0] Habana exception raised from LaunchRecipe at graph_exec.cpp:558
```

and are debugging.
See the similar error in [Example 3](#-example-3-graph-classification).

### ✅ Example 6: GNN Explanation

Adaptation completed.

See the [subfolder](Example6/).

### 🟡 Example 7: Aggregation Package

Currently, we cannot run `ClusterData`, which requires either `pyg-lib` or `torch-sparse`, while both `pyg-lib` and `torch-sparse` only support CUDA GPUs.
See the error messages [here](Example7/error.pdf).

**Workaround:** We are able to adapt the code by building the realted functions (specifically, METIS parition) in `torch_sparse` from C source files. See more details [here](../../funcs/metis_partition/).

See the [subfolder](Example7/).

See the related issue on Pytorch Cluster GitHub repo [here](https://github.com/rusty1s/pytorch_cluster/issues/230).

See the related discussion on PyG GitHub repo [here](https://github.com/pyg-team/pytorch_geometric/discussions/9760).

See the related question on Intel Gaudi forum [here](https://forum.habana.ai/t/functionalities-that-require-pyg-lib-torch-sparse-torch-cluster/1363).

### ✅ Example 8: Node Classification (with W&B)

Adaptation completed.

See the [subfolder](Example8/).

### 🟡 Example 9: Graph Classification with PyG and W&B

Currently, we encounter

```plaintext
RuntimeError: [Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in Lowering thread...
[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT Run returned exception....
synNodeCreateWithId failed for node: concat with synStatus 1 [Invalid argument]. .
[Rank:0] Habana exception raised from add_node at graph.cpp:507
[Rank:0] Habana exception raised from LaunchRecipe at graph_exec.cpp:558
```

and are debugging.
See the error messages [here](Example9/error.pdf).
See the similar error in [Example 3](#-example-3-graph-classification).

**Workaround:** We are able to adapt the code by removing `model = torch.compile(model, backend="hpu_backend")`.
See such adapted code [here](Example9/9_Graph_Classification_with_PyG_and_W&B_no_compile.ipynb).
See also the debugging information [below](#example-9-debugging-information).

See the [subfolder](Example9/).

See the related question on Intel Gaudi forum [here](https://forum.habana.ai/t/gcnconv-fails-with-normalization/1336).

#### Example 9: Debugging Information

Seemingly, the problem is caused by the function `gcn_norm` since it works well when we set `normalize = False` for `GCNConv`.
See the code without normalization [here](./Example9/9_Graph_Classification_with_PyG_and_W&B_no_normalize.ipynb).
After removing `model = torch.compile(model, backend="hpu_backend")`, it works even with `normalize = True`.

### 🟡 Example 10: Link Prediction on MovieLens

Currently, we cannot run `NeighborSampler`, which requires either `pyg-lib` or `torch-sparse`, while both `pyg-lib` and `torch-sparse` only support CUDA GPUs.
See the error messages [here](Example10/error.pdf).

**Workaround (NeighborSampler):** We are able to adapt the code by building the realted functions (specifically, "neighbor sample") in `torch_sparse` from C source files. See more details [here](../../funcs/neighbor_sample/).

**Workaround (GCNConv):** We are able to adapt the code by removing `model = torch.compile(model, backend="hpu_backend")`. See more details above in [Example 3](#-example-3-graph-classification).

See the [subfolder](Example10/).

See the related issue on Pytorch Cluster GitHub repo [here](https://github.com/rusty1s/pytorch_cluster/issues/230).

See the related discussion on PyG GitHub repo [here](https://github.com/pyg-team/pytorch_geometric/discussions/9760).

See the related question on Intel Gaudi forum [here](https://forum.habana.ai/t/functionalities-that-require-pyg-lib-torch-sparse-torch-cluster/1363).

### 🟡 Example 11: Link Regression on Movielens

Currently, we encounter `AttributeError: module 'habana_frameworks.torch.hpu' has no attribute 'wrap_in_hpu_graph'` and are debugging.

- The [Sentence Transformers](https://sbert.net/index.html) module tries to call `habana_frameworks.torch.hpu.wrap_in_hpu_graph`, which seemingly does not exist.

See the error messages [here](Example11/error.pdf).

**Workaround:** The function `wrap_in_hpu_graph` is in `habana_frameworks/torch/hpu/graphs.py` but the functions there are imported in `__init__.py` only when `is_lazy()` is True. We are able to adapt the code by manually modifying `habana_frameworks/torch/hpu/__init__.py` by removing the condition `if is_lazy()` and it works now.

See the [subfolder](Example11/).

See the related question on Intel Gaudi forum [here](https://forum.habana.ai/t/attributeerror-module-habana-frameworks-torch-hpu-has-no-attribute-wrap-in-hpu-graph/1362).

See the related question on Intel Gaudi forum [here](https://forum.habana.ai/t/gcnconv-fails-with-normalization/1336).
