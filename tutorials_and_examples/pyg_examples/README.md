# PyG Examples with Intel Gaudi-v2 Devices

The examples here are adapted from [PyG official examples](https://pytorch-geometric.readthedocs.io/en/stable/get_started/colabs.html).

## Contents

### ✅ Example 1: Introduction

Adaptation completed.
See the [subfolder](Example1/).

### ✅ Example 2: Node Classification

Adaptation completed.
See the [subfolder](Example2/).

### ✅ Example 3: Graph Classification

Adaptation completed.
See the [subfolder](Example3/).
Notes: The code is runnable only without `model = torch.compile(model, backend="hpu_backend")`.
Otherwise, we encounter

```plaintext
RuntimeError: [Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in Lowering thread...
[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT Run returned exception....
synNodeCreateWithId failed for node: concat with synStatus 1 [Invalid argument]. .
[Rank:0] Habana exception raised from add_node at graph.cpp:507
[Rank:0] Habana exception raised from LaunchRecipe at graph_exec.cpp:558
```

and are debugging.
Seemingly, the problem is caused by the function `gcn_norm` since it works well with `model = torch.compile(model, backend="hpu_backend")` when we set `normalize = False` for `GCNConv`.
See the error messages [here](Example3/error.pdf).

### ❌ Example 4: Scaling GNNs

Currently, we cannot run `ClusterData`, which requires either `pyg-lib` or `torch-sparse`, while both `pyg-lib` and `torch-sparse` only support CUDA GPUs.
See the error messages [here](Example4/error.pdf).
See the [subfolder](Example4/).

### ❌ Example 5: Point Cloud Classification

Currently, we cannot run `knn_graph`, which requires `torch-cluster`, while `torch-cluster` only supports CUDA GPUs.
See the error messages [here](Example5/error.pdf).
See the [subfolder](Example5/).

### ✅ Example 6: GNN Explanation

Adaptation completed.
See the [subfolder](Example6/).

### ❌ Example 7: Aggregation Package

Currently, we cannot run `ClusterData`, which requires either `pyg-lib` or `torch-sparse`, while both `pyg-lib` and `torch-sparse` only support CUDA GPUs.
See the error messages [here](Example7/error.pdf).
See the [subfolder](Example7/).

### ✅ Example 8: Node Classification (with W&B)

Adaptation completed.
See the [subfolder](Example8/).

### ❌ Example 9: Graph Classification with PyG and W&B

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
See the [subfolder](Example9/).

### ❌ Example 10: Link Prediction on MovieLens

Currently, we cannot run `NeighborSampler`, which requires either `pyg-lib` or `torch-sparse`, while both `pyg-lib` and `torch-sparse` only support CUDA GPUs.
See the error messages [here](Example10/error.pdf).
See the [subfolder](Example10/).

### ❌ Example 11: Link Regression on Movielens

Currently, we encounter `AttributeError: module 'habana_frameworks.torch.hpu' has no attribute 'wrap_in_hpu_graph'` and are debugging.

- The [Sentence Transformers](https://sbert.net/index.html) module tries to call `habana_frameworks.torch.hpu.wrap_in_hpu_graph`, which seemingly does not exist.

See the error messages [here](Example11/error.pdf).
See the [subfolder](Example11/).
