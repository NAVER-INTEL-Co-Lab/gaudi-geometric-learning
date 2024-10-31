# PyG Examples with Intel Gaudi-v2 Devices

The examples here are adapted from [PyG official examples](https://pytorch-geometric.readthedocs.io/en/stable/get_started/colabs.html).

## Contents

### ✅ Example 1: Introduction

Adaptation completed.
See the [subfolder](Example1/).

### ✅ Example 2: Node Classification

Adaptation completed.
See the [subfolder](Example2/).

### ❌ Example 3: Graph Classification

Currently, we encounter

```plaintext
RuntimeError: [Rank:0] FATAL ERROR :: MODULE:PT_BRIDGE Exception in Lowering thread...
[Rank:0] FATAL ERROR :: MODULE:PT_EAGER HabanaLaunchOpPT Run returned exception....
synNodeCreateWithId failed for node: concat with synStatus 1 [Invalid argument]. .
[Rank:0] Habana exception raised from add_node at graph.cpp:507
[Rank:0] Habana exception raised from LaunchRecipe at graph_exec.cpp:558
```

and are debugging.
Seemingly, the problem is caused by the function `gcn_norm` since it works when we set `normalize = False` for `GCNConv`.
See the error messages [here](Example3/error.pdf).
See the [subfolder](Example3/).

### ❌ Example 4: Scaling GNNs

Currently, we cannot run `ClusterData`, which requires either `pyg-lib` or `torch-sparse`, while both `pyg-lib` and `torch-sparse` only support CUDA GPUs.
See the error messages [here](Example4/error.pdf).
See the [subfolder](Example4/).
