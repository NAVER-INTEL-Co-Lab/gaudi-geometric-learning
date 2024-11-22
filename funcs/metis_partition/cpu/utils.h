#pragma once

#include "../extensions.h"

#define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")
#define CHECK_LT(low, high) AT_ASSERTM(low < high, "low must be smaller than high")

#define AT_DISPATCH_HAS_VALUE(optional_value, ...)                             \
  [&] {                                                                        \
    if (optional_value.has_value()) {                                          \
      const bool HAS_VALUE = true;                                             \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      const bool HAS_VALUE = false;                                            \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

inline int64_t uniform_randint(int64_t low, int64_t high) {
  CHECK_LT(low, high);
  auto options = torch::TensorOptions().dtype(torch::kInt64);
  auto ret = torch::randint(low, high, {1}, options);
  auto ptr = ret.data_ptr<int64_t>();
  return *ptr;
}

inline int64_t uniform_randint(int64_t high) {
  return uniform_randint(0, high);
}

inline torch::Tensor
choice(int64_t population, int64_t num_samples, bool replace = false,
       torch::optional<torch::Tensor> weight = torch::nullopt) {

  if (population == 0 || num_samples == 0)
    return torch::empty({0}, at::kLong);

  if (!replace && num_samples >= population)
    return torch::arange(population, at::kLong);

  if (weight.has_value())
    return torch::multinomial(weight.value(), num_samples, replace);

  if (replace) {
    const auto out = torch::empty({num_samples}, at::kLong);
    auto *out_data = out.data_ptr<int64_t>();
    for (int64_t i = 0; i < num_samples; i++) {
      out_data[i] = uniform_randint(population);
    }
    return out;

  } else {
    // Sample without replacement via Robert Floyd algorithm:
    // https://www.nowherenearithaca.com/2013/05/
    // robert-floyds-tiny-and-beautiful.html
    const auto out = torch::empty({num_samples}, at::kLong);
    auto *out_data = out.data_ptr<int64_t>();
    std::unordered_set<int64_t> samples;
    for (int64_t i = population - num_samples; i < population; i++) {
      int64_t sample = uniform_randint(i);
      if (!samples.insert(sample).second) {
        sample = i;
        samples.insert(sample);
      }
      out_data[i - population + num_samples] = sample;
    }
    return out;
  }
}

