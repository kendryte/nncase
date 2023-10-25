#include "runtime_utils.h"
#include <apply.h>
#include <cmath>
#ifdef __riscv_vector_
#include <riscv_vector.h>
#endif

using namespace nncase::runtime::xpu;
namespace kernels {

namespace {
template <typename T>
void instance_norm_naive_impl(
    const T *input, const T *sum, T *sum_sqr, T *output, T *gamma, T *beta,
    gsl::span<const size_t> input_shape, gsl::span<const size_t> input_stride,
    gsl::span<const size_t> output_stride,
    [[maybe_unused]] gsl::span<const size_t> sum_strides,
    gsl::span<const size_t> gamma_strides, T eps, int32_t norm_size) noexcept {
    auto axis = 2;
    apply(input_shape, [&](gsl::span<const size_t> input_index) -> void {
        //  input_index
        auto o_offset = offset(sum_strides, input_index.subspan(0, axis));
        auto mean = sum[o_offset] / norm_size;
        auto sigma = nncase_mt->float_unary_sqrt(sum_sqr[o_offset] / norm_size -
                                                 mean * mean + eps);

        auto input_offset = offset(input_stride, input_index);
        auto in_offset =
            offset(gamma_strides, input_index.subspan(axis - 1, 1));
        output[offset(output_stride, input_index)] =
            (input[input_offset] - mean) / sigma * gamma[in_offset] +
            beta[in_offset];
    });
}
} // namespace

template <class T>
void instance_norm(const T *input, T *sum, T *sum_sqr, T *output, T *gamma,
                   T *beta, dims_t input_dims, strides_t input_strides,
                   strides_t output_strides, strides_t sum_strides,
                   strides_t gamma_strides, T eps, int32_t norm_size) {
    return instance_norm_naive_impl(
        input, sum, sum_sqr, output, gamma, beta,
        gsl::make_span(input_dims).template as_span<const size_t>(),
        gsl::make_span(input_strides).template as_span<const size_t>(),
        gsl::make_span(output_strides).template as_span<const size_t>(),
        gsl::make_span(sum_strides).template as_span<const size_t>(),
        gsl::make_span(gamma_strides).template as_span<const size_t>(), eps,
        norm_size);
}
} // namespace kernels