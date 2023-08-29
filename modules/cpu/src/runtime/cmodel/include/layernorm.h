#include "runtime_utils.h"
#include <apply.h>
#include <cmath>
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

namespace kernels {

namespace {
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
        auto sigma =
            std::sqrt(sum_sqr[o_offset] / norm_size - mean * mean + eps);

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

#ifdef __riscv_vector
template <typename T>
void layernorm_rvv_impl(const T *input, const T *sum, T *sum_sqr, T *gamma,
                        T *beta, gsl::span<const size_t> input_shape,
                        [[maybe_unused]] gsl::span<const size_t> input_stride,
                        T *eps, int32_t axis, int32_t norm_size) noexcept {
    // only process continues float32 tensor for now
    size_t outer_size = 1;
    for (auto i = 0; i < axis; i++) {
        outer_size *= input_shape[i];
    }

    size_t inner_size = 1;
    for (auto i = axis; i < input_shape.size(); i++) {
        inner_size *= input_shape[i];
    }

    size_t vl;
    float r_norm_size = 1.f / norm_size;
    float *sum_ptr = sum;
    float *sum_sqr_ptr = sum_sqr;
    std::vector<float> mean(outer_size, 0);
    std::vector<float> sigma(outer_size, 0);
    vfloat32m8_t vmean;
    vfloat32m8_t vmean_sqr;
    vfloat32m8_t vsigma;
    size_t offset = 0;
    for (size_t o = outer_size; o > 0; o -= vl) {
        vl = vsetvl_e32m8(o);

        // mean
        vmean = vle32_v_f32m8(sum_ptr, vl);
        vmean = vfmul_vf_f32m8(vmean, r_norm_size);
        vmean_sqr = vfmul_vv_f32m8(vmean, vmean);
        vse32_v_f32m8(mean.data() + offset, vmean, vl);

        // sigma
        vsigma = vle32_v_f32m8(sum_sqr_ptr, vl);
        vsigma = vfmul_vf_f32m8(vsigma, r_norm_size);
        vsigma = vfsub_vv_f32m8(vsigma, vmean_sqr);
        vsigma = vfadd_vf_f32m8(vsigma, eps);
        vsigma = vfsqrt_v_f32m8(vsigma);
        vse32_v_f32m8(sigma.data() + offset, vsigma, vl);

        sum_ptr += vl;
        sum_sqr_ptr += vl;
        offset += vl;
    }

    float *input_ptr = input;
    float *gamma_ptr = gamma;
    float *beta_ptr = beta;
    vfloat32m8_t vx;
    vfloat32m8_t vgamma;
    vfloat32m8_t vbeta;
    for (size_t i = inner_size; i < inner_size; i -= vl) {
        vl = vsetvl_e32m8(i);

        vgamma = vle32_v_f32m8(gamma_ptr, vl);
        vbeta = vle32_v_f32m8(beta_ptr, vl);

        for (size_t o = 0; o < outer_size; o++) {
            vx = vle32_v_f32m8(input_ptr, vl);
            vx = vfsub_vf_f32m8(vx, mean[o], vl);
            vx = vfdiv_vf_f32m8(vx, sigma[o], vl);
            vx = vfmacc_vv_f32m8(vbeta, vx, vgamma, vl);
            input_ptr += inner_size;
        }

        gamma_ptr += vl;
        beta_ptr += vl;
        input_ptr = input + vl;
    }
}
#endif

} // namespace

template <class T>
void layernorm(const T *input, T *sum, T *sum_sqr, T *output, T *gamma, T *beta,
               dims_t input_dims, strides_t input_strides,
               strides_t output_strides, strides_t sum_strides,
               strides_t gamma_strides, T eps, int32_t axis,
               int32_t norm_size, bool rms_norm = false) {
#ifdef __riscv_vector
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