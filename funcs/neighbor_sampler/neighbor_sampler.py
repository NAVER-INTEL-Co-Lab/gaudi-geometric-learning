# torch_geometric/loader/neighbor_sampler.py

from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor

# from torch_geometric.typing import SparseTensor
import random


class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


# class Adj(NamedTuple):
#     adj_t: SparseTensor
#     e_id: Optional[Tensor]
#     size: Tuple[int, int]

#     def to(self, *args, **kwargs):
#         adj_t = self.adj_t.to(*args, **kwargs)
#         e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
#         return Adj(adj_t, e_id, self.size)


class DenseNeighborSampler(torch.utils.data.DataLoader):
    r"""The neighbor sampler from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, which allows
    for mini-batch training of GNNs on large-scale graphs where full-batch
    training is not feasible.

    Given a GNN with :math:`L` layers and a specific mini-batch of nodes
    :obj:`node_idx` for which we want to compute embeddings, this module
    iteratively samples neighbors and constructs bipartite graphs that simulate
    the actual computation flow of GNNs.

    More specifically, :obj:`sizes` denotes how much neighbors we want to
    sample for each node in each layer.
    This module then takes in these :obj:`sizes` and iteratively samples
    :obj:`sizes[l]` for each node involved in layer :obj:`l`.
    In the next layer, sampling is repeated for the union of nodes that were
    already encountered.
    The actual computation graphs are then returned in reverse-mode, meaning
    that we pass messages from a larger set of nodes to a smaller one, until we
    reach the nodes for which we originally wanted to compute embeddings.

    Hence, an item returned by :class:`NeighborSampler` holds the current
    :obj:`batch_size`, the IDs :obj:`n_id` of all nodes involved in the
    computation, and a list of bipartite graph objects via the tuple
    :obj:`(edge_index, e_id, size)`, where :obj:`edge_index` represents the
    bipartite edges between source and target nodes, :obj:`e_id` denotes the
    IDs of original edges in the full graph, and :obj:`size` holds the shape
    of the bipartite graph.
    For each bipartite graph, target nodes are also included at the beginning
    of the list of source nodes so that one can easily apply skip-connections
    or add self-loops.

    .. warning::

        :class:`~torch_geometric.loader.NeighborSampler` is deprecated and will
        be removed in a future release.
        Use :class:`torch_geometric.loader.NeighborLoader` instead.

    .. note::

        For an example of using :obj:`NeighborSampler`, see
        `examples/reddit.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        reddit.py>`_ or
        `examples/ogbn_products_sage.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        ogbn_products_sage.py>`_.

    Args:
        edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
            :class:`torch_sparse.SparseTensor` that defines the underlying
            graph connectivity/message passing flow.
            :obj:`edge_index` holds the indices of a (sparse) symmetric
            adjacency matrix.
            If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its shape
            must be defined as :obj:`[2, num_edges]`, where messages from nodes
            :obj:`edge_index[0]` are sent to nodes in :obj:`edge_index[1]`
            (in case :obj:`flow="source_to_target"`).
            If :obj:`edge_index` is of type :class:`torch_sparse.SparseTensor`,
            its sparse indices :obj:`(row, col)` should relate to
            :obj:`row = edge_index[1]` and :obj:`col = edge_index[0]`.
            The major difference between both formats is that we need to input
            the *transposed* sparse adjacency matrix.
        sizes ([int]): The number of neighbors to sample for each node in each
            layer. If set to :obj:`sizes[l] = -1`, all neighbors are included
            in layer :obj:`l`.
        node_idx (LongTensor, optional): The nodes that should be considered
            for creating mini-batches. If set to :obj:`None`, all nodes will be
            considered.
        num_nodes (int, optional): The number of nodes in the graph.
            (default: :obj:`None`)
        return_e_id (bool, optional): If set to :obj:`False`, will not return
            original edge indices of sampled edges. This is only useful in case
            when operating on graphs without edge features to save memory.
            (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(self, edge_index: Tensor,
                 sizes: List[int], node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None, return_e_id: bool = True,
                 transform: Callable = None, **kwargs):

        edge_index = edge_index.to('cpu')

        # Remove for PyTorch Lightning:
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        # Save for Pytorch Lightning < 1.6:
        self.edge_index = edge_index
        self.node_idx = node_idx
        self.num_nodes = num_nodes

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform
        # self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        self.__val__ = None

        # Obtain a *transposed* `SparseTensor` instance.        
        if (num_nodes is None and node_idx is not None
                and node_idx.dtype == torch.bool):
            num_nodes = node_idx.size(0)
        if (num_nodes is None and node_idx is not None
                and node_idx.dtype == torch.long):
            num_nodes = max(int(edge_index.max()), int(node_idx.max())) + 1
        if num_nodes is None:
            num_nodes = int(edge_index.max()) + 1
        
        self.num_nodes = num_nodes
        self.value = torch.arange(edge_index.size(1)) if return_e_id else None
        self.row = edge_index[0]
        self.col = edge_index[1]
        self.sparse_sizes = (num_nodes, num_nodes)               
        
        self.adj_list = [[] for _ in range(num_nodes)]
        self.edge_ids = [[] for _ in range(num_nodes)]
        
        for e_id, (src, dst) in enumerate(edge_index.t()):
            self.adj_list[src.item()].append(dst.item())
            if return_e_id:
                self.edge_ids[src.item()].append(e_id)
        
        if node_idx is None:
            node_idx = torch.arange(num_nodes)
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        super().__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def sample_neighbors(self, nodes: Tensor, size: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample neighbors for given nodes"""
        rows, cols, e_ids = [], [], []
        
        for node in nodes.tolist():
            neighbors = self.adj_list[node]
            if len(neighbors) > 0:
                if size > 0:
                    num_samples = min(size, len(neighbors))
                    sampled_idx = random.sample(range(len(neighbors)), num_samples)
                    sampled_neighbors = [neighbors[i] for i in sampled_idx]
                    if self.return_e_id:
                        sampled_e_ids = [self.edge_ids[node][i] for i in sampled_idx]
                else:
                    sampled_neighbors = neighbors
                    sampled_e_ids = self.edge_ids[node] if self.return_e_id else []
                
                rows.extend([node] * len(sampled_neighbors))
                cols.extend(sampled_neighbors)
                if self.return_e_id:
                    e_ids.extend(sampled_e_ids)
        
        edge_index = torch.tensor([cols, rows], dtype=torch.long)
        e_id = torch.tensor(e_ids) if self.return_e_id else None
        return edge_index, e_id


    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        adjs = []
        n_id = batch
        for size in self.sizes:
            # adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            edge_index, e_id = self.sample_neighbors(n_id, size)
            
            # Get new nodes involved
            new_nodes = torch.unique(edge_index[0])
            n_id = torch.unique(torch.cat([n_id, new_nodes]))
            
            size = (n_id.size(0), len(batch))
            adjs.append(EdgeIndex(edge_index, e_id, size))

        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (batch_size, n_id, adjs)
        out = self.transform(*out) if self.transform is not None else out
        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sizes={self.sizes})'
