#include <Python.h>
#include <torch/script.h>
#include "cpu/metis_cpu.h"

#ifdef _WIN32
#ifdef WITH_PYTHON
PyMODINIT_FUNC PyInit__metis_cpu(void) { return NULL; }
#endif
#endif

SPARSE_API torch::Tensor partition(torch::Tensor rowptr, torch::Tensor col,
                        torch::optional<torch::Tensor> optional_value,
                        int64_t num_parts, bool recursive) {  
    return partition_cpu(rowptr, col, optional_value, torch::nullopt, num_parts,
                         recursive);  
}

SPARSE_API torch::Tensor partition2(torch::Tensor rowptr, torch::Tensor col,
                         torch::optional<torch::Tensor> optional_value,
                         torch::optional<torch::Tensor> optional_node_weight,
                         int64_t num_parts, bool recursive) {  
    return partition_cpu(rowptr, col, optional_value, optional_node_weight,
                         num_parts, recursive);
}

static auto registry = torch::RegisterOperators()
                           .op("torch_sparse::partition", &partition)
                           .op("torch_sparse::partition2", &partition2);
