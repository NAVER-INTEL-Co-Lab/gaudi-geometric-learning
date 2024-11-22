import torch
from typing import Optional, Tuple


# torch_scatter/utils.py
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


# torch_scatter/scatter.py
def scatter_sum_raw(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


# torch_scatter/scatter.py
def scatter_add_raw(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    return scatter_sum_raw(src, index, dim, out, dim_size)


# torch_scatter/scatter.py
def scatter_mean_raw(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    out = scatter_sum_raw(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum_raw(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out


# torch_scatter/scatter.py
# value only, equivalent to scatter_max(...)[0]
def scatter_max_raw(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_reduce(
            0, index=index, src=src, reduce="amax", include_self=False
        )

    else:
        return out.scatter_reduce(
            0, index=index, src=src, reduce="amax", include_self=False
        )


# torch_scatter/scatter.py
# value only, equivalent to scatter_min(...)[0]
def scatter_min_raw(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_reduce(
            0, index=index, src=src, reduce="amin", include_self=False
        )

    else:
        return out.scatter_reduce(
            0, index=index, src=src, reduce="amin", include_self=False
        )


# torch_scatter/scatter.py
def scatter_mul_raw(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:

    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.ones(size, dtype=src.dtype, device=src.device)
        return out.scatter_reduce_(
            0, index=index, src=src, reduce="prod", include_self=False
        )
    else:
        return out.scatter_reduce_(
            0, index=index, src=src, reduce="prod", include_self=False
        )


# torch_scatter/composite/softmax.py
def scatter_softmax_raw(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError(
            "`scatter_softmax` can only be computed over tensors "
            "with floating point data types."
        )

    index = broadcast(index, src, dim)

    max_value_per_index = scatter_max_raw(src, index, dim=dim, dim_size=dim_size)
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp_()

    sum_per_index = scatter_sum_raw(
        recentered_scores_exp, index, dim, dim_size=dim_size
    )
    normalizing_constants = sum_per_index.gather(dim, index)

    return recentered_scores_exp.div(normalizing_constants)


# torch_scatter/composite/softmax.py
def scatter_log_softmax_raw(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-12,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError(
            "`scatter_log_softmax` can only be computed over "
            "tensors with floating point data types."
        )

    index = broadcast(index, src, dim)

    max_value_per_index = scatter_max_raw(src, index, dim=dim, dim_size=dim_size)
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element

    sum_per_index = scatter_sum_raw(
        recentered_scores.exp(), index, dim, dim_size=dim_size
    )
    normalizing_constants = sum_per_index.add_(eps).log_().gather(dim, index)

    return recentered_scores.sub_(normalizing_constants)


# torch_scatter/composite/logsumexp.py
def scatter_logsumexp_raw(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError(
            "`scatter_logsumexp` can only be computed over "
            "tensors with floating point data types."
        )

    index = broadcast(index, src, dim)

    if out is not None:
        dim_size = out.size(dim)
    else:
        if dim_size is None:
            dim_size = int(index.max()) + 1

    size = list(src.size())
    size[dim] = dim_size
    max_value_per_index = torch.full(
        size, float("-inf"), dtype=src.dtype, device=src.device
    )
    max_value_per_index = scatter_max_raw(
        src, index, dim, max_value_per_index, dim_size=dim_size
    )
    max_per_src_element = max_value_per_index.gather(dim, index)
    recentered_score = src - max_per_src_element
    recentered_score.masked_fill_(torch.isnan(recentered_score), float("-inf"))

    if out is not None:
        out = out.sub_(max_value_per_index).exp_()

    sum_per_index = scatter_sum_raw(recentered_score.exp_(), index, dim, out, dim_size)

    out = sum_per_index.add_(eps).log_().add_(max_value_per_index)
    return out.nan_to_num_(neginf=0.0)


# torch_scatter/composite/std.py
def scatter_std_raw(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    unbiased: bool = True,
) -> torch.Tensor:

    if out is not None:
        dim_size = out.size(dim)

    if dim < 0:
        dim = src.dim() + dim

    count_dim = dim
    if index.dim() <= dim:
        count_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum_raw(ones, index, count_dim, dim_size=dim_size)

    index = broadcast(index, src, dim)
    tmp = scatter_sum_raw(src, index, dim, dim_size=dim_size)
    count = broadcast(count, tmp, dim).clamp(1)
    mean = tmp.div(count)

    var = src - mean.gather(dim, index)
    var = var * var
    out = scatter_sum_raw(var, index, dim, out, dim_size)

    if unbiased:
        count = count.sub(1).clamp_(1)
    out = out.div(count + 1e-6).sqrt()

    return out
