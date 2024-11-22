#include <Python.h>
#include <torch/script.h>
#include "cpu/knn_cpu.h"

#ifdef _WIN32
PyMODINIT_FUNC PyInit__knn_cpu(void) { return NULL; }
#endif

CLUSTER_API torch::Tensor knn(torch::Tensor x, torch::Tensor y,
                  torch::optional<torch::Tensor> ptr_x,
                  torch::optional<torch::Tensor> ptr_y, int64_t k, bool cosine,
                  int64_t num_workers) {
  if (cosine)
    AT_ERROR("`cosine` argument not supported on CPU");
  return knn_cpu(x, y, ptr_x, ptr_y, k, num_workers);
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::knn", &knn);
