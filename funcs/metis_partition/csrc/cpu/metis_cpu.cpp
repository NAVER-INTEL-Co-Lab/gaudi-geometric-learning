#include "metis_cpu.h"
#include <metis.h>
#include "utils.h"

torch::Tensor partition_cpu(torch::Tensor rowptr, torch::Tensor col,
                            torch::optional<torch::Tensor> optional_value,
                            torch::optional<torch::Tensor> optional_node_weight,
                            int64_t num_parts, bool recursive) {
#ifdef WITH_METIS
  CHECK_CPU(rowptr);
  CHECK_CPU(col);

  if (optional_value.has_value()) {
    CHECK_CPU(optional_value.value());
    CHECK_INPUT(optional_value.value().dim() == 1);
    CHECK_INPUT(optional_value.value().numel() == col.numel());
  }

  if (optional_node_weight.has_value()) {
    CHECK_CPU(optional_node_weight.value());
    CHECK_INPUT(optional_node_weight.value().dim() == 1);
    CHECK_INPUT(optional_node_weight.value().numel() == rowptr.numel() - 1);
  }

  int64_t nvtxs = rowptr.numel() - 1;
  int64_t ncon = 1;
  auto *xadj = rowptr.data_ptr<int64_t>();
  auto *adjncy = col.data_ptr<int64_t>();

  int64_t *adjwgt = NULL;
  if (optional_value.has_value())
    adjwgt = optional_value.value().data_ptr<int64_t>();

  int64_t *vwgt = NULL;
  if (optional_node_weight.has_value())
    vwgt = optional_node_weight.value().data_ptr<int64_t>();

  int64_t objval = -1;
  auto part = torch::empty({nvtxs}, rowptr.options());
  auto part_data = part.data_ptr<int64_t>();

  if (recursive) {
    METIS_PartGraphRecursive(&nvtxs, &ncon, xadj, adjncy, vwgt, NULL, adjwgt,
                             &num_parts, NULL, NULL, NULL, &objval, part_data);
  } else {
    METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, NULL, adjwgt,
                        &num_parts, NULL, NULL, NULL, &objval, part_data);
  }

  return part;
#else
  AT_ERROR("Not compiled with METIS support");
#endif
}