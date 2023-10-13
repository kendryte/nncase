#include "runtime_utils.h"
#include <apply.h>
#include <cmath>
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

using namespace nncase::runtime::cpu;
namespace kernels {

namespace {

#ifdef __riscv_vector
static void layernor_restruct2(const float* input, float* output, int len, const float* gamma, const float* beta,
 float mean, float sigma)
{   
    size_t vl;
    if(gamma == NULL && beta == NULL)
    {
        for (size_t i = len; i > 0; i -= vl) {
            vl = vsetvl_e32m8(i);
            vfloat32m8_t vx = vle32_v_f32m8(input, vl);
            vx = vfsub_vf_f32m8(vx, mean, vl);
            vx = vfdiv_vf_f32m8(vx, sigma, vl);
            vse32_v_f32m8(output, vx, vl);
            input += vl;
            output += vl;
        }
        return;
    }
    if(gamma == NULL)
    {
        for (size_t i = len; i > 0; i -= vl) {
            vl = vsetvl_e32m8(i);
            vfloat32m8_t vx = vle32_v_f32m8(input, vl);
            vfloat32m8_t vbeta = vle32_v_f32m8(beta, vl);
            vx = vfsub_vf_f32m8(vx, mean, vl);
            vx = vfdiv_vf_f32m8(vx, sigma, vl);
            vx = vfadd_vv_f32m8(vx, vbeta, vl);
            vse32_v_f32m8(output, vx, vl);
            input += vl;
            output += vl;
            beta += vl;
        }
        return;
    }
    if(beta == NULL)
    {
        for (size_t i = len; i > 0; i -= vl) {
            vl = vsetvl_e32m8(i);
            vfloat32m8_t vx = vle32_v_f32m8(input, vl);
            vfloat32m8_t vgamma = vle32_v_f32m8(gamma, vl);
            vx = vfsub_vf_f32m8(vx, mean, vl);
            vx = vfdiv_vf_f32m8(vx, sigma, vl);
            vx = vfmul_vv_f32m8(vx, vgamma, vl);
            vse32_v_f32m8(output, vx, vl);
            input += vl;
            output += vl;
            gamma += vl;
        }
        return;
    }
    
    for (size_t i = len; i > 0; i -= vl) {
        vl = vsetvl_e32m8(i);
        vfloat32m8_t vx = vle32_v_f32m8(input, vl);
        vfloat32m8_t vgamma = vle32_v_f32m8(gamma, vl);
        vfloat32m8_t vbeta = vle32_v_f32m8(beta, vl);
        vx = vfsub_vf_f32m8(vx, mean, vl);
        vx = vfdiv_vf_f32m8(vx, sigma, vl);
        vx = vfmacc_vv_f32m8(vbeta, vx, vgamma, vl);
        vse32_v_f32m8(output, vx, vl);
        input += vl;
        output += vl;
        beta += vl;
        gamma += vl;
    }
}

static int  get_offset_from_index(gsl::span<const size_t> in_shape, gsl::span<const size_t> in_strides, int index)
{
    strides_t x = get_default_strides(in_shape);
    int __sum = 0;
    for(size_t i = 0; i < x.size(); ++i)
    {
        __sum += index / x[i] * in_strides[i];
        index = index % x[i];
    }
    return __sum;
}

template <typename T>
void layernorm_naive_impl(const T *input, const T *sum, const T *sum_sqr, T *output,
                          const T *gamma, const T *beta,
                          gsl::span<const size_t> input_shape,
                          gsl::span<const size_t> input_stride,
                          gsl::span<const size_t> output_stride,
                          [[maybe_unused]] gsl::span<const size_t> sum_strides,
                          [[maybe_unused]] gsl::span<const size_t> gamma_strides, T eps,
                          int32_t axis, int32_t norm_size,
                          [[maybe_unused]] bool rms_norm = false) noexcept {

    size_t outer_size = 1;
    for (auto i = 0; i < axis; i++) {
        outer_size *= input_shape[i];
    }

    size_t inner_size = 1;
    for (int i = axis; i < (int)input_shape.size(); i++) {
        inner_size *= input_shape[i];
    }
    gsl::span<const size_t> in_shape_outer(input_shape.begin(), input_shape.begin() + axis);
    gsl::span<const size_t> in_shape_inner(input_shape.begin() + axis, input_shape.end());
    for (size_t o = 0; o < outer_size; o++) {
        int __ptr_output = get_offset_from_index(in_shape_outer, output_stride, o);
        int __ptr_input = get_offset_from_index(in_shape_outer, input_stride, o);
        int __ptr_sum = get_offset_from_index(in_shape_outer, sum_strides, o);
        auto mean = sum[__ptr_sum] / norm_size;
        if(rms_norm) mean = 0;
        auto sigma = std::sqrt(sum_sqr[__ptr_sum] / norm_size - mean * mean + eps);
        layernor_restruct2(input + __ptr_input, output + __ptr_output, inner_size, gamma, beta, mean, sigma);
    }
}
#else
template <typename T>
void layernorm_naive_impl(const T *input, const T *sum, T *sum_sqr, T *output,
                          T *gamma, T *beta,
                          gsl::span<const size_t> input_shape,
                          gsl::span<const size_t> input_stride,
                          gsl::span<const size_t> output_stride,
                          [[maybe_unused]] gsl::span<const size_t> sum_strides,
                          gsl::span<const size_t> gamma_strides, T eps,
                          int32_t axis, int32_t norm_size,
                          bool rms_norm = false) noexcept {
    apply(input_shape, [&](gsl::span<const size_t> input_index) -> void {
        //  input_index
        auto o_offset = offset(sum_strides, input_index.subspan(0, axis));
        // auto o_offset = input_index[0];
        auto mean = sum[o_offset] / norm_size;
        if (rms_norm) {
            mean = 0;
        }
        auto sigma = nncase_mt->float_unary_sqrt(sum_sqr[o_offset] / norm_size -
                                                 mean * mean + eps);

        auto input_offset = offset(input_stride, input_index);
        auto in_offset = offset(gamma_strides, input_index.subspan(axis));
        output[offset(output_stride, input_index)] =
            (input[input_offset] - mean) / sigma *
                (gamma == nullptr ? static_cast<T>(1) : gamma[in_offset]) +
            (beta == nullptr ? static_cast<T>(0) : beta[in_offset]);
    });

    // // only process continues tensor for now
    // size_t outer_size = 1;
    // for (auto i = 0; i < axis; i++) {
    //     outer_size *= input_shape[i];
    // }

    // size_t inner_size = 1;
    // for (auto i = axis; i < input_shape.size(); i++) {
    //     inner_size *= input_shape[i];
    // }

    // for (size_t o = 0; o < outer_size; o++) {
    //     auto mean = sum[o] / norm_size;
    //     auto sigma = std::sqrt(sum_sqr[o] / norm_size - mean * mean + eps);
    //     for (size_t i = 0; i < inner_size; i++) {
    //         auto x = input + o * inner_size + i;
    //         *x = (*x - mean) / sigma *
    //                  (gamma == nullptr ? static_cast<T>(1) : gamma[i]) +
    //              (beta == nullptr ? static_cast<T>(0) : beta[i]);
    //     }
    // }
}
#endif

} // namespace

template <class T>
void layernorm(const T *input, T *sum, T *sum_sqr, T *output, T *gamma, T *beta,
               dims_t input_dims, strides_t input_strides,
               strides_t output_strides, strides_t sum_strides,
               strides_t gamma_strides, T eps, int32_t axis, int32_t norm_size,
               bool rms_norm = false) {
                // fafdaf
#ifdef __riscv_vector_
    return layernorm_rvv_impl(
        input, sum, sum_sqr, gamma, beta,
        gsl::make_span(input_dims).template as_span<const size_t>(),
        gsl::make_span(input_strides).template as_span<const size_t>(), eps,
        axis, norm_size);
#else
    return layernorm_naive_impl(
        input, sum, sum_sqr, output, gamma, beta,
        gsl::make_span(input_dims).template as_span<const size_t>(),
        gsl::make_span(input_strides).template as_span<const size_t>(),
        gsl::make_span(output_strides).template as_span<const size_t>(),
        gsl::make_span(sum_strides).template as_span<const size_t>(),
        gsl::make_span(gamma_strides).template as_span<const size_t>(), eps,
        axis, norm_size, rms_norm);
#endif
}
} // namespace kernels