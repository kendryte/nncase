#include "runtime_utils.h"
#include <apply.h>
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

using namespace nncase::runtime::xpu;
namespace kernels {

namespace {
#ifdef __riscv_vector
static void instance_norm_restruct2(const float *input, float *output, int len,
                                    const float gamma, const float beta,
                                    float mean, float sigma) {
    size_t vl;
    for (size_t i = len; i > 0; i -= vl) {
        vl = vsetvl_e32m8(i);
        vfloat32m8_t vx = vle32_v_f32m8(input, vl);
        vx = vfsub_vf_f32m8(vx, mean, vl);
        vx = vfdiv_vf_f32m8(vx, sigma, vl);
        vx = vfmul_vf_f32m8(vx, gamma, vl);
        vx = vfadd_vf_f32m8(vx, beta, vl);
        vse32_v_f32m8(output, vx, vl);
        input += vl;
        output += vl;
    }
}

template <typename T>
void instance_norm_rvv_impl(
    const T *input, const T *sum, const T *sum_sqr, T *output, const T *gamma,
    const T *beta, gsl::span<const size_t> input_shape,
    gsl::span<const size_t> input_stride, gsl::span<const size_t> output_stride,
    [[maybe_unused]] gsl::span<const size_t> sum_strides,
    [[maybe_unused]] gsl::span<const size_t> gamma_strides, T eps,
    int32_t norm_size) noexcept {

    auto get_offset_from_index = [](gsl::span<const size_t> in_shape,
                                    gsl::span<const size_t> in_strides,
                                    int index) -> int {
        strides_t x = get_default_strides(in_shape);
        int __sum = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            __sum += index / x[i] * in_strides[i];
            index = index % x[i];
        }
        return __sum;
    };

    size_t outer_size = 1;
    for (auto i = 0; i < 2; i++) {
        outer_size *= input_shape[i];
    }

    size_t inner_size = 1;
    for (int i = 2; i < (int)input_shape.size(); i++) {
        inner_size *= input_shape[i];
    }
    gsl::span<const size_t> in_shape_outer(input_shape.begin(),
                                           input_shape.begin() + 2);
    gsl::span<const size_t> in_shape_inner(input_shape.begin() + 2,
                                           input_shape.end());
    for (size_t o = 0; o < outer_size; o++) {
        int __ptr_output =
            get_offset_from_index(in_shape_outer, output_stride, o);
        int __ptr_input =
            get_offset_from_index(in_shape_outer, input_stride, o);
        int __ptr_sum = get_offset_from_index(in_shape_outer, sum_strides, o);
        auto mean = sum[__ptr_sum] / norm_size;
        auto sigma = nncase_mt->float_unary_sqrt(
            sum_sqr[__ptr_sum] / norm_size - mean * mean + eps);
        auto in_offset = outer_size % input_shape[1];
        instance_norm_restruct2(input + __ptr_input, output + __ptr_output,
                                inner_size, gamma[in_offset], beta[in_offset],
                                mean, sigma);
    }
}
#else
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
#endif

} // namespace

template <class T>
void instance_norm(const T *input, T *sum, T *sum_sqr, T *output, T *gamma,
                   T *beta, dims_t input_dims, strides_t input_strides,
                   strides_t output_strides, strides_t sum_strides,
                   strides_t gamma_strides, T eps, int32_t norm_size) {
#ifdef __riscv_vector
    return instance_norm_rvv_impl(
        input, sum, sum_sqr, output, gamma, beta,
        gsl::make_span(input_dims).template as_span<const size_t>(),
        gsl::make_span(input_strides).template as_span<const size_t>(),
        gsl::make_span(output_strides).template as_span<const size_t>(),
        gsl::make_span(sum_strides).template as_span<const size_t>(),
        gsl::make_span(gamma_strides).template as_span<const size_t>(), eps,
        norm_size);
#else
    return instance_norm_naive_impl(
        input, sum, sum_sqr, output, gamma, beta,
        gsl::make_span(input_dims).template as_span<const size_t>(),
        gsl::make_span(input_strides).template as_span<const size_t>(),
        gsl::make_span(output_strides).template as_span<const size_t>(),
        gsl::make_span(sum_strides).template as_span<const size_t>(),
        gsl::make_span(gamma_strides).template as_span<const size_t>(), eps,
        norm_size);
#endif
}
} // namespace kernels