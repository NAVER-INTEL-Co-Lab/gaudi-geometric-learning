#include <Python.h>
#include <torch/script.h>

#include "cpu/rw_cpu.h"

#ifdef _WIN32
PyMODINIT_FUNC PyInit__rw_cpu(void) { return NULL; }
#endif

CLUSTER_API std::tuple<torch::Tensor, torch::Tensor>
random_walk(torch::Tensor rowptr, torch::Tensor col, torch::Tensor start,
            int64_t walk_length, double p, double q) {
    return random_walk_cpu(rowptr, col, start, walk_length, p, q);  
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::random_walk", &random_walk);
