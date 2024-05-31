/* Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "../reference/ref_ops.h"
#include "nncase/runtime/util.h"
#include "opt_ops.h"
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <utility>
#ifdef NNCASE_HALIDE
#include <hkg/export/HalideBuffer.h>
#include <hkg/export/halide_conv2d.h>
#include <hkg/export/halide_conv2d_depthwise.h>
#endif
#ifdef NNCASE_OPENMP
#include <omp.h>
#endif

#define CONV_ARGS                                                              \
    input, weights, bias, output, in_shape, in_strides, w_shape, w_strides,    \
        bias_strides, out_strides, padding_h, padding_w, groups, stride_h,     \
        stride_w, dilation_h, dilation_w, fused_activation, context

#define CONV2D_NXM_S1_S2(n, m)                                                 \
    if (filter_h == n && filter_w == m) {                                      \
        if (stride_h == 1 && stride_w == 1) {                                  \
            return conv2d_nxm<8, n, m, 1, 1>(CONV_ARGS);                       \
        } else if (stride_h == 2 && stride_w == 2) {                           \
            return conv2d_nxm<8, n, m, 2, 2>(CONV_ARGS);                       \
        }                                                                      \
    }

#define CONV2D_DEPTHWISE_NXM_S1_S2(n, m)                                       \
    if (filter_h == n && filter_w == m) {                                      \
        if (stride_h == 1 && stride_w == 1) {                                  \
            return conv2d_depthwise_nxm<8, n, m, 1, 1>(CONV_ARGS);             \
        } else if (stride_h == 2 && stride_w == 2) {                           \
            return conv2d_depthwise_nxm<8, n, m, 2, 2>(CONV_ARGS);             \
        }                                                                      \
    }

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;

template <typename T>
result<void>
conv2d_1x1_s1(const T *input, const T *weights, const T *bias, T *output,
              std::span<const size_t> in_shape,
              NNCASE_UNUSED std::span<const size_t> in_strides,
              NNCASE_UNUSED std::span<const size_t> w_shape,
              NNCASE_UNUSED std::span<const size_t> w_strides,
              NNCASE_UNUSED std::span<const size_t> bias_strides,
              NNCASE_UNUSED std::span<const size_t> out_strides,
              NNCASE_UNUSED const padding &padding_h,
              NNCASE_UNUSED const padding &padding_w,
              NNCASE_UNUSED int32_t groups, NNCASE_UNUSED int32_t stride_h,
              NNCASE_UNUSED int32_t stride_w, NNCASE_UNUSED int32_t dilation_h,
              NNCASE_UNUSED int32_t dilation_w, value_range<T> fused_activation,
              NNCASE_UNUSED kernels::kernel_context &context) noexcept {
    const auto widths = in_shape[2] * in_shape[3];
    // if oc's type is size_t, openmp will throw error in visual studio
    // if no cast, compiler will throw warning because of comparison of integer
    // expressions of different signedness warning be treated as errors
    const auto out_channels = w_shape[0];

    for (size_t batch = 0; batch < in_shape[0]; batch++) {
#ifdef NNCASE_OPENMP
#pragma omp parallel for num_threads(context.num_threads)
#endif
        for (int oc = 0; oc < out_channels; oc++) {
            const auto out_c = oc;
            const T *now_weights = weights + out_c * w_strides[0];
            const T *now_img_start = input + batch * in_strides[0];
            size_t channel = 0;

            auto *now_output_channel_start =
                output + (batch * out_strides[0] + out_c * out_strides[1]);

            std::fill(now_output_channel_start,
                      now_output_channel_start + in_shape[2] * in_shape[3],
                      bias[oc]);
            for (; channel + 4 <= in_shape[1]; channel += 4, now_weights += 4) {
                auto *w_output = now_output_channel_start;
                const T w0 = now_weights[0];
                const T w1 = now_weights[1];
                const T w2 = now_weights[2];
                const T w3 = now_weights[3];

                const T *i0 = now_img_start + (channel + 0) * in_strides[1];
                const T *i1 = now_img_start + (channel + 1) * in_strides[1];
                const T *i2 = now_img_start + (channel + 2) * in_strides[1];
                const T *i3 = now_img_start + (channel + 3) * in_strides[1];

                const T *v0 = i0;
                const T *v1 = i1;
                const T *v2 = i2;
                const T *v3 = i3;

                for (size_t index = 0; index < widths; ++index) {
                    T sum0 = *v0 * w0;
                    T sum1 = *v1 * w1;
                    T sum2 = *v2 * w2;
                    T sum3 = *v3 * w3;

                    *w_output += sum0 + sum1 + sum2 + sum3;

                    ++w_output;
                    ++v0;
                    ++v1;
                    ++v2;
                    ++v3;
                }
            }

            for (; channel < in_shape[1]; ++channel) {
                auto *w_output = now_output_channel_start;
                const T *v = now_img_start + channel * in_strides[1];
                for (size_t index = 0; index < widths; ++index) {
                    *w_output += (T)(*now_weights) * (T)(*v);
                    ++w_output;
                    ++v;
                }
                ++now_weights;
            }

            for (size_t i = 0; i < widths; i++) {
                *(now_output_channel_start + i) =
                    kernels::detail::apply_activation(
                        *(now_output_channel_start + i), fused_activation);
            }
        }
    }
    return ok();
}

template <typename T>
result<void>
conv2d_1x1_s2(const T *input, const T *weights, const T *bias, T *output,
              std::span<const size_t> in_shape,
              NNCASE_UNUSED std::span<const size_t> in_strides,
              NNCASE_UNUSED std::span<const size_t> w_shape,
              NNCASE_UNUSED std::span<const size_t> w_strides,
              NNCASE_UNUSED std::span<const size_t> bias_strides,
              NNCASE_UNUSED std::span<const size_t> out_strides,
              NNCASE_UNUSED const padding &padding_h,
              NNCASE_UNUSED const padding &padding_w,
              NNCASE_UNUSED int32_t groups, NNCASE_UNUSED int32_t stride_h,
              NNCASE_UNUSED int32_t stride_w, NNCASE_UNUSED int32_t dilation_h,
              NNCASE_UNUSED int32_t dilation_w, value_range<T> fused_activation,
              NNCASE_UNUSED kernels::kernel_context &context) noexcept {
    const auto batch = in_shape[0], in_channels = in_shape[1],
               in_h = in_shape[2], in_w = in_shape[3],
               out_channels = w_shape[0];
    const auto filter_h = (int32_t)w_shape[2];
    const auto filter_w = (int32_t)w_shape[3];
    const auto out_h = nncase::kernels::detail::get_windowed_output_size(
        in_h, filter_h, stride_h, dilation_h, padding_h);
    const auto out_w = nncase::kernels::detail::get_windowed_output_size(
        in_w, filter_w, stride_w, dilation_w, padding_w);

    const size_t tailstep = in_w - (out_w * stride_w);

    for (size_t b = 0; b < batch; b++) {
#ifdef NNCASE_OPENMP
#pragma omp parallel for num_threads(context.num_threads)
#endif
        for (int oc = 0; oc < out_channels; oc++) {
            T *out = output + (b * out_strides[0] + oc * out_strides[1]);

            std::fill(out, out + out_h * out_w, bias[oc]);
            size_t ic = 0;
            for (; ic + 3 < in_channels; ic += 4) {
                T *outptr = out;
                const T *img0 =
                    input + (b * in_strides[0]) + (ic * in_strides[1]);
                const T *img1 =
                    input + (b * in_strides[0]) + ((ic + 1) * in_strides[1]);
                const T *img2 =
                    input + (b * in_strides[0]) + ((ic + 2) * in_strides[1]);
                const T *img3 =
                    input + (b * in_strides[0]) + ((ic + 3) * in_strides[1]);

                const T *r0 = img0;
                const T *r1 = img1;
                const T *r2 = img2;
                const T *r3 = img3;

                const T *k0 = weights + oc * w_strides[0] + ic * w_strides[1];
                const T *k1 = k0 + 1;
                const T *k2 = k0 + 2;
                const T *k3 = k0 + 3;
                for (size_t i = 0; i < out_h; i++) {
                    for (size_t remain = 0; remain < out_w; remain++) {
                        *outptr += r0[0] * k0[0];
                        *outptr += r1[0] * k1[0];
                        *outptr += r2[0] * k2[0];
                        *outptr += r3[0] * k3[0];
                        r0 += 2;
                        r1 += 2;
                        r2 += 2;
                        r3 += 2;
                        outptr++;
                    }
                    r0 += tailstep + in_w;
                    r1 += tailstep + in_w;
                    r2 += tailstep + in_w;
                    r3 += tailstep + in_w;
                }
            }

            for (; ic < in_channels; ic++) {
                T *outptr = out;
                const T *img0 =
                    input + (b * in_strides[0]) + (ic * in_strides[1]);
                const T *kernel0 =
                    weights + oc * w_strides[0] + ic * w_strides[1];
                const T *r0 = img0;
                const T *k0 = kernel0;
                for (size_t i = 0; i < out_h; i++) {
                    for (size_t remain = 0; remain < out_w; remain++) {
                        *outptr += r0[0] * k0[0];
                        r0 += 2;
                        outptr++;
                    }
                    r0 += tailstep + in_w;
                }
            }
            for (size_t h = 0; h < out_h; h++) {
                T *r_out = out + h * out_strides[2];
                for (size_t w = 0; w < out_w; w++) {
                    *(r_out + w) = kernels::detail::apply_activation(
                        *(r_out + w), fused_activation);
                }
            }
        }
    }
    return ok();
}

template <size_t Parallel, size_t Stride, size_t Filter>
constexpr size_t compute_rsize() {
    return Filter + std::min(Stride, Filter) * (Parallel - 1);
}

template <typename T, size_t N, size_t... I>
void binding_ptr_unfold(std::array<T, N> &a, T base, size_t step, size_t start,
                        std::index_sequence<I...>) {
    NNCASE_UNUSED int dummy[] = {0, (a[start + I] = base + step * I, 0)...};
}

template <typename T, size_t... W>
void conv_1xm(T &sum, const T *r, const T *k, std::index_sequence<W...>) {
    NNCASE_UNUSED int dummy[] = {0, (sum += r[W] * k[W], 0)...};
}

template <size_t R, size_t Filter_h, size_t Filter_w, typename T, size_t N,
          size_t... H>
void conv_nxm_unfold_n(T &sum, std::array<const T *, N> &r,
                       std::array<const T *, Filter_h> &k,
                       std::index_sequence<H...>) {
    NNCASE_UNUSED int dummy[] = {
        0, (conv_1xm(sum, r[R + H], k[H], std::make_index_sequence<Filter_w>{}),
            0)...};
}

template <size_t R, size_t Filter_h, size_t Filter_w, typename T, size_t N>
void conv_nxm(T &sum, std::array<const T *, N> &r,
              std::array<const T *, Filter_h> &k) {
    conv_nxm_unfold_n<R, Filter_h, Filter_w>(
        sum, r, k, std::make_index_sequence<Filter_h>{});
}

template <typename T, size_t N, size_t... I>
void binding_value_unfold(std::array<T *, N> &a, std::array<T, N> &b,
                          std::index_sequence<I...>) {

    NNCASE_UNUSED int dummy[] = {0, (*a[I] += b[I], 0)...};
}

template <typename Array, size_t... I>
void increase_n_unfold(Array &a, size_t step, std::index_sequence<I...>) {
    NNCASE_UNUSED int dummy[] = {0, (a[I] += step, 0)...};
}

template <size_t N, typename Array>
void increase_n_dispatch(NNCASE_UNUSED Array &a, NNCASE_UNUSED size_t step,
                         std::false_type) {}

template <size_t N, typename Array>
void increase_n_dispatch(Array &a, size_t step, std::true_type) {
    increase_n_unfold(a, step, std::make_index_sequence<N>{});
}

template <size_t Parallel, size_t P, size_t Stride_h, size_t Filter_h,
          size_t Filter_w, typename T, size_t N, size_t X>
void conv_nxm(NNCASE_UNUSED std::array<T, X> &sum,
              NNCASE_UNUSED std::array<const T *, N> &r,
              NNCASE_UNUSED std::array<const T *, Filter_h> &k,
              std::false_type) {}

template <size_t Parallel, size_t P, size_t Stride_h, size_t Filter_h,
          size_t Filter_w, typename T, size_t N, size_t X>
void conv_nxm(std::array<T, X> &sum, std::array<const T *, N> &r,
              std::array<const T *, Filter_h> &k, std::true_type) {
    conv_nxm<P * std::min(Stride_h, Filter_h), Filter_h, Filter_w>(sum[P], r,
                                                                   k);
    conv_nxm<Parallel, P + 1, Stride_h, Filter_h, Filter_w, T, N, X>(
        sum, r, k,
        std::integral_constant<bool, std::less<size_t>()(P + 1, Parallel)>{});
}

template <size_t Filter, typename T, size_t N>
void binding_ptr(std::array<T, N> &a, T base, size_t step, size_t start = 0) {
    binding_ptr_unfold(a, base, step, start,
                       std::make_index_sequence<Filter>{});
}

template <size_t Stride, size_t Filter, typename T, size_t N, size_t... P>
void binding_ptr_unfold(std::array<T, N> &a, T base, size_t step,
                        std::index_sequence<P...>) {
    NNCASE_UNUSED int dummy[] = {
        0, (binding_ptr<Filter>(a, base + P * Stride * step, step,
                                P * std::min(Filter, Stride)),
            0)...};
}

template <size_t Parallel, size_t Stride, size_t Filter, typename T, size_t N>
void binding_ptr(std::array<T, N> &a, T base, size_t step) {
    binding_ptr_unfold<Stride, Filter>(a, base, step,
                                       std::make_index_sequence<Parallel>());
}

template <size_t Parallel, typename T, size_t N>
void binding_value(std::array<T *, N> &output, std::array<T, N> &value) {
    binding_value_unfold(output, value, std::make_index_sequence<Parallel>{});
}

template <size_t N, typename Array> void increase_n(Array &a, size_t step = 1) {
    increase_n_dispatch<N>(
        a, step, std::integral_constant<bool, std::greater<size_t>()(N, 0)>{});
}

template <size_t LocalParallel, size_t Filter_h, size_t Filter_w,
          size_t Stride_h, size_t Stride_w, typename T, size_t R,
          size_t Parallel>
void conv2d_channel_dispatch(NNCASE_UNUSED size_t &i,
                             NNCASE_UNUSED size_t out_h,
                             NNCASE_UNUSED size_t out_w,
                             NNCASE_UNUSED std::array<T, Parallel> &sum,
                             NNCASE_UNUSED std::array<const T *, R> &r,
                             NNCASE_UNUSED std::array<const T *, Filter_h> k,
                             NNCASE_UNUSED std::array<T *, Parallel> outptr,
                             NNCASE_UNUSED size_t in_w_step,
                             NNCASE_UNUSED size_t out_w_step,
                             NNCASE_UNUSED size_t tail_step, std::false_type) {}

template <size_t LocalParallel, size_t Filter_h, size_t Filter_w,
          size_t Stride_h, size_t Stride_w, typename T, size_t R,
          size_t Parallel>
void conv2d_channel_dispatch(size_t &i, size_t out_h, size_t out_w,
                             std::array<T, Parallel> &sum,
                             std::array<const T *, R> &r,
                             std::array<const T *, Filter_h> k,
                             std::array<T *, Parallel> outptr, size_t in_w_step,
                             size_t out_w_step, size_t tail_step,
                             std::true_type) {
    for (; i + (LocalParallel - 1) < out_h; i += LocalParallel) {
        for (size_t remain = 0; remain < out_w; remain++) {
            std::fill_n(sum.begin(), LocalParallel, static_cast<T>(0));
            conv_nxm<LocalParallel, 0, Stride_h, Filter_h, Filter_w>(
                sum, r, k, std::true_type{});
            binding_value<LocalParallel>(outptr, sum);
            increase_n<compute_rsize<LocalParallel, Stride_h, Filter_h>()>(
                r, Stride_w);
            increase_n<LocalParallel>(outptr, 1);
        }
        increase_n<compute_rsize<LocalParallel, Stride_h, Filter_h>()>(
            r, (Stride_h * LocalParallel - 1) * in_w_step + tail_step);
        increase_n<LocalParallel>(outptr, out_w_step * (LocalParallel - 1));
    }
    conv2d_channel_dispatch<LocalParallel / 2, Filter_h, Filter_w, Stride_h,
                            Stride_w>(
        i, out_h, out_w, sum, r, k, outptr, in_w_step, out_w_step, tail_step,
        std::integral_constant<bool,
                               std::greater<size_t>()(LocalParallel, 1)>{});
}

template <size_t LocalParallel, size_t Filter_h, size_t Filter_w,
          size_t Stride_h, size_t Stride_w, typename T, size_t R,
          size_t Parallel>
void conv2d_channel(size_t out_h, size_t out_w, std::array<T, Parallel> &sum,
                    std::array<const T *, R> &r,
                    std::array<const T *, Filter_h> k,
                    std::array<T *, Parallel> outptr, size_t in_w_step,
                    size_t out_w_step, size_t tail_step) {
    size_t i = 0;
    conv2d_channel_dispatch<LocalParallel, Filter_h, Filter_w, Stride_h,
                            Stride_w>(i, out_h, out_w, sum, r, k, outptr,
                                      in_w_step, out_w_step, tail_step,
                                      std::true_type{});
}

template <size_t Parallel, size_t Filter_h, size_t Filter_w, size_t Stride_h,
          size_t Stride_w, typename T>
result<void>
conv2d_nxm(const T *input, const T *weights, const T *bias, float *output,
           std::span<const size_t> in_shape,
           NNCASE_UNUSED std::span<const size_t> in_strides,
           NNCASE_UNUSED std::span<const size_t> w_shape,
           NNCASE_UNUSED std::span<const size_t> w_strides,
           NNCASE_UNUSED std::span<const size_t> bias_strides,
           NNCASE_UNUSED std::span<const size_t> out_strides,
           NNCASE_UNUSED const padding &padding_h,
           NNCASE_UNUSED const padding &padding_w, NNCASE_UNUSED int32_t groups,
           NNCASE_UNUSED int32_t stride_h, NNCASE_UNUSED int32_t stride_w,
           NNCASE_UNUSED int32_t dilation_h, NNCASE_UNUSED int32_t dilation_w,
           value_range<float> fused_activation,
           NNCASE_UNUSED kernels::kernel_context &context) noexcept {
    const auto batch = in_shape[0], out_channels = w_shape[0],
               in_channels = w_shape[1], in_h = in_shape[2], in_w = in_shape[3];
    const auto out_h = nncase::kernels::detail::get_windowed_output_size(
        in_h, Filter_h, Stride_h, dilation_h, padding::zero());
    const auto out_w = nncase::kernels::detail::get_windowed_output_size(
        in_w, Filter_w, Stride_w, dilation_w, padding::zero());
    const size_t tail_step = in_strides[2] - (out_w * Stride_w);
    for (size_t b = 0; b < batch; b++) // batch
    {
#ifdef NNCASE_OPENMP
#pragma omp parallel for num_threads(context.num_threads)
#endif
        for (int oc = 0; oc < out_channels; oc++) // out channel
        {
            std::array<float *, Parallel> outptr;
            std::array<const float *,
                       compute_rsize<Parallel, Stride_h, Filter_h>()>
                r;
            std::array<const float *, Filter_h> k;
            std::array<float, Parallel> sum;

            float *out = output + out_strides[0] * b + out_strides[1] * oc;
            std::fill_n(out,
                        out_strides[2]
                            ? out_h * out_strides[2]
                            : (out_strides[3] ? out_w * out_strides[3] : 1),
                        bias[oc]); // avoid shape == 1, stride == 0

            for (size_t ic = 0; ic < in_channels; ic++) // in channel
            {
                binding_ptr<Parallel>(outptr, out, out_strides[2]);
                binding_ptr<Parallel, Stride_h, Filter_h>(
                    r, input + in_strides[0] * b + in_strides[1] * ic,
                    in_strides[2]);
                binding_ptr<Filter_h>(
                    k, weights + w_strides[0] * oc + w_strides[1] * ic,
                    w_strides[2]);
                conv2d_channel<Parallel, Filter_h, Filter_w, Stride_h,
                               Stride_w>(out_h, out_w, sum, r, k, outptr,
                                         in_strides[2], out_strides[2],
                                         tail_step);
            }
            for (size_t h = 0; h < out_h; h++) {
                float *r_out = out + h * out_strides[2];
                for (size_t w = 0; w < out_w; w++) {
                    *(r_out + w) = kernels::detail::apply_activation(
                        *(r_out + w), fused_activation);
                }
            }
        }
    }
    return ok();
}

template <size_t Parallel, size_t Filter_h, size_t Filter_w, size_t Stride_h,
          size_t Stride_w, typename T>
result<void> conv2d_depthwise_nxm(
    const T *input, const T *weights, const T *bias, T *output,
    std::span<const size_t> in_shape,
    NNCASE_UNUSED std::span<const size_t> in_strides,
    NNCASE_UNUSED std::span<const size_t> w_shape,
    NNCASE_UNUSED std::span<const size_t> w_strides,
    NNCASE_UNUSED std::span<const size_t> bias_strides,
    NNCASE_UNUSED std::span<const size_t> out_strides,
    NNCASE_UNUSED const padding &padding_h,
    NNCASE_UNUSED const padding &padding_w, NNCASE_UNUSED int32_t groups,
    NNCASE_UNUSED int32_t stride_h, NNCASE_UNUSED int32_t stride_w,
    NNCASE_UNUSED int32_t dilation_h, NNCASE_UNUSED int32_t dilation_w,
    value_range<T> fused_activation,
    NNCASE_UNUSED kernels::kernel_context &context) noexcept {
    const auto batch = in_shape[0], channels = w_shape[0], in_h = in_shape[2],
               in_w = in_shape[3];
    const auto out_h = nncase::kernels::detail::get_windowed_output_size(
        in_h, Filter_h, Stride_h, dilation_h, padding::zero());
    const auto out_w = nncase::kernels::detail::get_windowed_output_size(
        in_w, Filter_w, Stride_w, dilation_w, padding::zero());

    const size_t tail_step = in_strides[2] - (out_w * Stride_w);
    for (size_t b = 0; b < batch; b++) // batch
    {

#ifdef NNCASE_OPENMP
#pragma omp parallel for num_threads(context.num_threads)
#endif
        for (int c = 0; c < channels; c++) // channel
        {
            std::array<T *, Parallel> outptr;
            std::array<const T *, compute_rsize<Parallel, Stride_h, Filter_h>()>
                r;
            std::array<const T *, Filter_h> k;
            std::array<T, Parallel> sum;

            T *out = output + out_strides[0] * b + out_strides[1] * c;
            std::fill_n(out,
                        out_strides[2]
                            ? out_h * out_strides[2]
                            : (out_strides[3] ? out_w * out_strides[3] : 1),
                        bias[c]);

            binding_ptr<Parallel>(outptr, out, out_strides[2]);
            binding_ptr<Parallel, Stride_h, Filter_h>(
                r, input + in_strides[0] * b + in_strides[1] * c,
                in_strides[2]);
            binding_ptr<Filter_h>(k, weights + w_strides[0] * c, w_strides[2]);
            conv2d_channel<Parallel, Filter_h, Filter_w, Stride_h, Stride_w>(
                out_h, out_w, sum, r, k, outptr, in_strides[2], out_strides[2],
                tail_step);
            for (size_t h = 0; h < out_h; h++) {
                T *r_out = out + h * out_strides[2];
                for (size_t w = 0; w < out_w; w++) {
                    *(r_out + w) = kernels::detail::apply_activation(
                        *(r_out + w), fused_activation);
                }
            }
        }
    }
    return ok();
}

#ifdef NNCASE_HALIDE
#define HALIDE_CONV2D_NXM_S1_S2(KH, KW)                                        \
    if (filter_h == (KH) && filter_w == (KW)) {                                \
        const auto out_h = nncase::kernels::detail::get_windowed_output_size(  \
            in_shape[2], filter_h, stride_h, dilation_h, padding_h);           \
        const auto out_w = nncase::kernels::detail::get_windowed_output_size(  \
            in_shape[3], filter_w, stride_w, dilation_w, padding_w);           \
        Halide::Runtime::Buffer<float> _input_buffer(                          \
            const_cast<float *>(input), in_shape[3], in_shape[2], in_shape[1], \
            in_shape[0]);                                                      \
        Halide::Runtime::Buffer<float> _weights_buffer(                        \
            const_cast<float *>(weights), w_shape[3], w_shape[2], w_shape[1],  \
            w_shape[0]);                                                       \
        Halide::Runtime::Buffer<float> _bias_buffer(const_cast<float *>(bias), \
                                                    w_shape[0]);               \
        float v_range[2] = {fused_activation.min, fused_activation.max};       \
        Halide::Runtime::Buffer<float> _value_range_buffer(v_range, 2);        \
        Halide::Runtime::Buffer<float> _Clamped_buffer(                        \
            output, out_w, out_h, w_shape[0], in_shape[0]);                    \
        if (stride_h == stride_w && (stride_h == 1 || stride_h == 2)) {        \
            halide_conv2d_##KH##x##KW(_input_buffer, _weights_buffer,          \
                                      _bias_buffer, _value_range_buffer,       \
                                      padding_h.before, padding_h.after,       \
                                      padding_w.before, padding_w.after,       \
                                      stride_h, stride_w, _Clamped_buffer);    \
            return ok();                                                       \
        }                                                                      \
    }

#define HALIDE_CONV2D_DEPTHWISE_NXM_S1_S2(KH, KW)                              \
    if (filter_h == (KH) && filter_w == (KW)) {                                \
        const auto out_h = nncase::kernels::detail::get_windowed_output_size(  \
            in_shape[2], filter_h, stride_h, dilation_h, padding_h);           \
        const auto out_w = nncase::kernels::detail::get_windowed_output_size(  \
            in_shape[3], filter_w, stride_w, dilation_w, padding_w);           \
        Halide::Runtime::Buffer<float> _input_buffer(                          \
            const_cast<float *>(input), in_shape[3], in_shape[2], in_shape[1], \
            in_shape[0]);                                                      \
        Halide::Runtime::Buffer<float> _weights_buffer(                        \
            const_cast<float *>(weights), w_shape[3], w_shape[2], w_shape[1],  \
            w_shape[0]);                                                       \
        Halide::Runtime::Buffer<float> _bias_buffer(const_cast<float *>(bias), \
                                                    w_shape[0]);               \
        float v_range[2] = {fused_activation.min, fused_activation.max};       \
        Halide::Runtime::Buffer<float> _value_range_buffer(v_range, 2);        \
        Halide::Runtime::Buffer<float> _Clamped_buffer(                        \
            output, out_w, out_h, w_shape[0], in_shape[0]);                    \
        if (stride_h == stride_w && (stride_h == 1 || stride_h == 2)) {        \
            halide_conv2d_depthwise_##KH##x##KW(                               \
                _input_buffer, _weights_buffer, _bias_buffer,                  \
                _value_range_buffer, padding_h.before, padding_h.after,        \
                padding_w.before, padding_w.after, stride_h, stride_w,         \
                _Clamped_buffer);                                              \
            return ok();                                                       \
        }                                                                      \
    }

#endif

result<void> optimized::conv2d(
    [[maybe_unused]] typecode_t typecode, const std::byte *input1,
    const std::byte *weights1, const std::byte *bias1, std::byte *output1,
    std::span<const size_t> in_shape, std::span<const size_t> in_strides,
    std::span<const size_t> w_shape,
    NNCASE_UNUSED std::span<const size_t> w_strides,
    NNCASE_UNUSED std::span<const size_t> bias_strides,
    NNCASE_UNUSED std::span<const size_t> out_strides, const padding &padding_h,
    const padding &padding_w, int32_t groups, int32_t stride_h,
    int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
    value_range<float> fused_activation,
    NNCASE_UNUSED kernels::kernel_context &context) noexcept {
    [[maybe_unused]] auto input = IN_CAST(float, input1);
    [[maybe_unused]] auto weights = IN_CAST(float, weights1);
    [[maybe_unused]] auto bias = IN_CAST(float, bias1);
    [[maybe_unused]] auto output = OUT_CAST(float, output1);
    const auto filter_h = w_shape[2];
    const auto filter_w = w_shape[3];

#ifdef NNCASE_HALIDE
    if (groups == 1 && runtime::is_contiguous(in_shape, in_strides)) {
        // clang-format off
        HALIDE_CONV2D_NXM_S1_S2(1, 1)
        else HALIDE_CONV2D_NXM_S1_S2(3, 3)
        else HALIDE_CONV2D_NXM_S1_S2(5, 5)
        else HALIDE_CONV2D_NXM_S1_S2(7, 7)
        // clang-format on
    }

    if ((size_t)groups == in_shape[1] && (size_t)groups == w_shape[0] &&
        runtime::is_contiguous(in_shape, in_strides)) {
        // clang-format off
        HALIDE_CONV2D_DEPTHWISE_NXM_S1_S2(1, 1)
        else HALIDE_CONV2D_DEPTHWISE_NXM_S1_S2(3, 3)
        else HALIDE_CONV2D_DEPTHWISE_NXM_S1_S2(5, 5)
        else HALIDE_CONV2D_DEPTHWISE_NXM_S1_S2(7, 7)
        // clang-format on
    }

#else
    if (groups == 1 && padding_h.before == 0 && padding_h.after == 0 &&
        padding_w.before == 0 && padding_w.after == 0) {
        if (filter_h == 1 && filter_w == 1) {
            if (stride_h == 1 && stride_w == 1) {
                return conv2d_1x1_s1(CONV_ARGS);
            } else if (stride_h == 2 && stride_w == 2) {
                return conv2d_1x1_s2(CONV_ARGS);
            }
        }
        // clang-format off
        else CONV2D_NXM_S1_S2(1, 3)
        else CONV2D_NXM_S1_S2(3, 1) 
        else CONV2D_NXM_S1_S2(3, 3) 
        else CONV2D_NXM_S1_S2(5, 5) 
        else CONV2D_NXM_S1_S2(7, 7)
        // clang-format on
    }

    if ((size_t)groups == in_shape[1] && (size_t)groups == w_shape[0] &&
        padding_h.before == 0 && padding_h.after == 0 &&
        padding_w.before == 0 && padding_w.after == 0) {
        // clang-format off
        CONV2D_DEPTHWISE_NXM_S1_S2(1, 3)
        else CONV2D_DEPTHWISE_NXM_S1_S2(3, 1) 
        else CONV2D_DEPTHWISE_NXM_S1_S2(3, 3) 
        else CONV2D_DEPTHWISE_NXM_S1_S2(5, 5) 
        else CONV2D_DEPTHWISE_NXM_S1_S2(7, 7)
        // clang-format on
    }
#endif
    try_(nncase::kernels::stackvm::reference::conv2d(
        typecode, input1, weights1, bias1, output1, in_shape, in_strides,
        w_shape, w_strides, bias_strides, out_strides, padding_h, padding_w,
        groups, stride_h, stride_w, dilation_h, dilation_w, fused_activation));
    return ok();
}

// result<void> optimized::conv2d(
//     [[maybe_unused]] typecode_t typecode, const std::byte *input,
//     const std::byte *weights, const std::byte *bias, std::byte *output,
//     std::span<const size_t> in_shape, std::span<const size_t> in_strides,
//     std::span<const size_t> w_shape,
//     NNCASE_UNUSED std::span<const size_t> w_strides,
//     NNCASE_UNUSED std::span<const size_t> bias_strides,
//     NNCASE_UNUSED std::span<const size_t> out_strides, const padding
//     &padding_h, const padding &padding_w, int32_t groups, int32_t stride_h,
//     int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
//     value_range<float> fused_activation,
//     NNCASE_UNUSED kernels::kernel_context &context) noexcept {
//     auto a = conv2d_impl(
//         IN_CAST(float, input), IN_CAST(float, weights), IN_CAST(float, bias),
//         OUT_CAST(float, output), in_shape, in_strides, w_shape, w_strides,
//         bias_strides, out_strides, padding_h, padding_w, groups, stride_h,
//         stride_w, dilation_h, dilation_w, fused_activation, context);
//     try_(nncase::kernels::stackvm::reference::conv2d(
//         typecode, input, weights, bias, output, in_shape, in_strides,
//         w_shape, w_strides, bias_strides, out_strides, padding_h, padding_w,
//         groups, stride_h, stride_w, dilation_h, dilation_w,
//         fused_activation));
//     return ok();
// }
