/* Copyright 2020 Canaan Inc.
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
#pragma once
#include "runtime_types.h"
#include <nncase/kernels/kernel_context.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/stackvm/kernel_context.h>
#include <utility>
#ifdef NNCASE_OPENMP
#include <omp.h>
#define GET_NUM_THREADS std::is_convertible<nncase::runtime::stackvm::stackvm_kernel_context &, decltype(context)>::value ? static_cast<nncase::runtime::stackvm::stackvm_kernel_context &>(context).num_threads_ : 1
#endif

BEGIN_NS_NNCASE_KERNELS_CPU_OPT

template <size_t Parallel, size_t Stride, size_t Filter>
constexpr size_t compute_rsize()
{
    return Filter + std::min(Stride, Filter) * (Parallel - 1);
}

template <typename T, size_t N, size_t... I>
void binding_ptr(std::array<T, N> &a, T base, size_t step, size_t start, std::index_sequence<I...>)
{
    NNCASE_UNUSED int dummy[] = { 0, (a[start + I] = base + step * I, 0)... };
}

template <size_t Filter, typename T, size_t N>
void binding_ptr(std::array<T, N> &a, T base, size_t step, size_t start = 0)
{
    binding_ptr(a, base, step, start, std::make_index_sequence<Filter> {});
}

template <size_t Stride, size_t Filter, typename T, size_t N, size_t... P>
void binding_ptr(std::array<T, N> &a, T base, size_t step, std::index_sequence<P...>)
{
    // ();
    NNCASE_UNUSED int dummy[] = { 0, (binding_ptr<Filter>(a, base + P * Stride * step, step, P * std::min(Filter, Stride)), 0)... };
}

template <size_t Parallel, size_t Stride, size_t Filter, typename T, size_t N>
void binding_ptr(std::array<T, N> &a, T base, size_t step)
{
    binding_ptr<Stride, Filter>(a, base, step, std::make_index_sequence<Parallel>());
}

namespace impl
{

template <typename T, size_t... W>
void conv1xM(T &sum, const T *r, const T *k, std::index_sequence<W...>)
{
    NNCASE_UNUSED int dummy[] = { 0, (sum += r[W] * k[W], 0)... };
}

template <size_t R, size_t Filter_h, size_t Filter_w, typename T, size_t N,
    size_t... H>
void convNxM(T &sum, std::array<const T *, N> &r, std::array<const T *, Filter_h> &k,
    std::index_sequence<H...>)
{
    NNCASE_UNUSED int dummy[] = { 0, (conv1xM(sum, r[R + H], k[H], std::make_index_sequence<Filter_w> {}), 0)... };
}

template <size_t R, size_t Filter_h, size_t Filter_w, typename T, size_t N>
void convNxM(T &sum, std::array<const T *, N> &r, std::array<const T *, Filter_h> &k)
{
    convNxM<R, Filter_h, Filter_w>(sum, r, k, std::make_index_sequence<Filter_h> {});
}

template <typename T, size_t N, size_t... I>
void binding_value(std::array<T *, N> &a, std::array<T, N> &b, std::index_sequence<I...>)
{

    NNCASE_UNUSED int dummy[] = { 0, (*a[I] += b[I], 0)... };
}

template <typename Array, size_t... I>
void increase_n(Array &a, size_t step, std::index_sequence<I...>)
{
    NNCASE_UNUSED int dummy[] = { 0, (a[I] += step, 0)... };
}

template <size_t N, typename Array>
void increase_n(NNCASE_UNUSED Array &a, NNCASE_UNUSED size_t step, std::false_type)
{
}

template <size_t N, typename Array>
void increase_n(Array &a, size_t step, std::true_type)
{
    increase_n(a, step, std::make_index_sequence<N> {});
}

} // namespace impl

template <size_t Parallel, typename T, size_t N>
void binding_value(std::array<T *, N> &output, std::array<T, N> &value)
{
    impl::binding_value(output, value, std::make_index_sequence<Parallel> {});
}

template <size_t N, typename Array>
void increase_n(Array &a, size_t step = 1)
{
    impl::increase_n<N>(a, step, std::integral_constant<bool, std::isgreater(N, 0)> {});
}

template <size_t Parallel, size_t P, size_t Stride_h, size_t Filter_h,
    size_t Filter_w, typename T, size_t N, size_t X>
void convNxM(NNCASE_UNUSED std::array<T, X> &sum, NNCASE_UNUSED std::array<const T *, N> &r,
    NNCASE_UNUSED std::array<const T *, Filter_h> &k, std::false_type)
{
}

template <size_t Parallel, size_t P, size_t Stride_h, size_t Filter_h,
    size_t Filter_w, typename T, size_t N, size_t X>
void convNxM(std::array<T, X> &sum, std::array<const T *, N> &r,
    std::array<const T *, Filter_h> &k, std::true_type)
{
    impl::convNxM<P * std::min(Stride_h, Filter_h), Filter_h, Filter_w>(sum[P], r, k);
    convNxM<Parallel, P + 1, Stride_h, Filter_h, Filter_w, T, N, X>(sum, r, k, std::integral_constant<bool, std::isless(P + 1, Parallel)> {});
}

template <size_t LocalParallel, size_t Filter_h, size_t Filter_w, size_t Stride_h, size_t Stride_w, typename T, size_t R, size_t Parallel>
void conv2dChannel(NNCASE_UNUSED size_t &i, NNCASE_UNUSED size_t out_h, NNCASE_UNUSED size_t out_w, NNCASE_UNUSED std::array<T, Parallel> &sum, NNCASE_UNUSED std::array<const T *, R> &r, NNCASE_UNUSED std::array<const T *, Filter_h> k,
    NNCASE_UNUSED std::array<T *, Parallel> outptr, NNCASE_UNUSED size_t in_w_step, NNCASE_UNUSED size_t out_w_step, NNCASE_UNUSED size_t tail_step, std::false_type)
{
}

template <size_t LocalParallel, size_t Filter_h, size_t Filter_w, size_t Stride_h, size_t Stride_w, typename T, size_t R, size_t Parallel>
void conv2dChannel(size_t &i, size_t out_h, size_t out_w, std::array<T, Parallel> &sum, std::array<const T *, R> &r, std::array<const T *, Filter_h> k,
    std::array<T *, Parallel> outptr, size_t in_w_step, size_t out_w_step, size_t tail_step, std::true_type)
{
    for (; i + (LocalParallel - 1) < out_h; i += LocalParallel)
    {
        for (size_t remain = 0; remain < out_w; remain++)
        {
            std::fill_n(sum.begin(), LocalParallel, 0);
            convNxM<LocalParallel, 0, Stride_h, Filter_h, Filter_w>(sum, r, k, std::true_type {});
            binding_value<LocalParallel>(outptr, sum);
            increase_n<compute_rsize<LocalParallel, Stride_h, Filter_h>()>(r, Stride_w);
            increase_n<LocalParallel>(outptr, 1);
        }
        increase_n<compute_rsize<LocalParallel, Stride_h, Filter_h>()>(r,
            (Stride_h * LocalParallel - 1) * in_w_step + tail_step);
        increase_n<LocalParallel>(outptr, out_w_step * (LocalParallel - 1));
    }
    conv2dChannel<LocalParallel / 2, Filter_h, Filter_w, Stride_h, Stride_w>(i, out_h, out_w, sum,
        r, k, outptr, in_w_step, out_w_step, tail_step, std::integral_constant<bool, std::isgreater(LocalParallel, 1)> {});
}

template <size_t LocalParallel, size_t Filter_h, size_t Filter_w, size_t Stride_h, size_t Stride_w, typename T, size_t R, size_t Parallel>
void conv2dChannel(size_t out_h, size_t out_w, std::array<T, Parallel> &sum, std::array<const T *, R> &r, std::array<const T *, Filter_h> k,
    std::array<T *, Parallel> outptr, size_t in_w_step, size_t out_w_step, size_t tail_step)
{
    size_t i = 0;
    conv2dChannel<LocalParallel, Filter_h, Filter_w, Stride_h, Stride_w>(i, out_h, out_w, sum, r, k, outptr, in_w_step, out_w_step, tail_step, std::true_type {});
}

result<void> conv2d_1x1_s1(const float *input, const float *weights, const float *bias, float *output,
    const runtime_shape_t &in_shape, NNCASE_UNUSED const runtime_shape_t &in_strides, NNCASE_UNUSED const runtime_shape_t &w_shape,
    NNCASE_UNUSED const runtime_shape_t &w_strides, NNCASE_UNUSED const runtime_shape_t &bias_strides, NNCASE_UNUSED const runtime_shape_t &out_strides,
    NNCASE_UNUSED const padding &padding_h, NNCASE_UNUSED const padding &padding_w,
    NNCASE_UNUSED int32_t groups, NNCASE_UNUSED int32_t stride_h, NNCASE_UNUSED int32_t stride_w,
    NNCASE_UNUSED int32_t dilation_h, NNCASE_UNUSED int32_t dilation_w, value_range<float> fused_activation, NNCASE_UNUSED kernel_context &context) noexcept;

result<void> conv2d_1x1_s2(const float *input, const float *weights, const float *bias, float *output,
    const runtime_shape_t &in_shape, NNCASE_UNUSED const runtime_shape_t &in_strides, NNCASE_UNUSED const runtime_shape_t &w_shape,
    NNCASE_UNUSED const runtime_shape_t &w_strides, NNCASE_UNUSED const runtime_shape_t &bias_strides, NNCASE_UNUSED const runtime_shape_t &out_strides,
    NNCASE_UNUSED const padding &padding_h, NNCASE_UNUSED const padding &padding_w,
    NNCASE_UNUSED int32_t groups, NNCASE_UNUSED int32_t stride_h, NNCASE_UNUSED int32_t stride_w,
    NNCASE_UNUSED int32_t dilation_h, NNCASE_UNUSED int32_t dilation_w, value_range<float> fused_activation, NNCASE_UNUSED kernel_context &context) noexcept;

NNCASE_API template <size_t Parallel, size_t Filter_h, size_t Filter_w, size_t Stride_h, size_t Stride_w>
result<void> conv2d_NxM(const float *input, const float *weights, const float *bias, float *output,
    const runtime_shape_t &in_shape, NNCASE_UNUSED const runtime_shape_t &in_strides, NNCASE_UNUSED const runtime_shape_t &w_shape,
    NNCASE_UNUSED const runtime_shape_t &w_strides, NNCASE_UNUSED const runtime_shape_t &bias_strides, NNCASE_UNUSED const runtime_shape_t &out_strides,
    NNCASE_UNUSED const padding &padding_h, NNCASE_UNUSED const padding &padding_w,
    NNCASE_UNUSED int32_t groups, NNCASE_UNUSED int32_t stride_h, NNCASE_UNUSED int32_t stride_w,
    NNCASE_UNUSED int32_t dilation_h, NNCASE_UNUSED int32_t dilation_w, value_range<float> fused_activation, NNCASE_UNUSED kernel_context &context) noexcept
{
    const auto batch = in_shape[0], out_channels = w_shape[0], in_channels = w_shape[1], in_h = in_shape[2], in_w = in_shape[3];
    const auto out_h = kernels::detail::get_windowed_output_size(in_h, Filter_h, Stride_h, dilation_h, padding::zero());
    const auto out_w = kernels::detail::get_windowed_output_size(in_w, Filter_w, Stride_w, dilation_w, padding::zero());
    const size_t tail_step = in_strides[2] - (out_w * Stride_w);
    for (size_t b = 0; b < batch; b++) // batch
    {
#ifdef NNCASE_OPENMP
#pragma omp parallel for num_threads(GET_NUM_THREADS)
#endif
        for (size_t oc = 0; oc < out_channels; oc++) // out channel
        {
            std::array<float *, Parallel> outptr;
            std::array<const float *, compute_rsize<Parallel, Stride_h, Filter_h>()> r;
            std::array<const float *, Filter_h> k;
            std::array<float, Parallel> sum;

            float *out = output + out_strides[0] * b + out_strides[1] * oc;
            std::fill_n(out, out_strides[2] ? out_h * out_strides[2] : (out_strides[3] ? out_w * out_strides[3] : 1), bias[oc]); // avoid shape == 1, stride == 0

            for (size_t ic = 0; ic < in_channels; ic++) // in channel
            {
                binding_ptr<Parallel>(outptr, out, out_strides[2]);
                binding_ptr<Parallel, Stride_h, Filter_h>(r, input + in_strides[0] * b + in_strides[1] * ic, in_strides[2]);
                binding_ptr<Filter_h>(k, weights + w_strides[0] * oc + w_strides[1] * ic, w_strides[2]);
                conv2dChannel<Parallel, Filter_h, Filter_w, Stride_h, Stride_w>(out_h, out_w, sum, r, k, outptr, in_strides[2], out_strides[2], tail_step);
            }
        }
    }
    for (size_t _ = 0; _ < batch * out_channels * out_h * out_w; _++)
    {
        *(output + _) = kernels::detail::apply_activation(*(output + _), fused_activation);
    }
    return ok();
}

NNCASE_API template <size_t Parallel, size_t Filter_h, size_t Filter_w, size_t Stride_h, size_t Stride_w>
result<void> conv2ddepthwise_NxM(const float *input, const float *weights, const float *bias, float *output,
    const runtime_shape_t &in_shape, NNCASE_UNUSED const runtime_shape_t &in_strides, NNCASE_UNUSED const runtime_shape_t &w_shape,
    NNCASE_UNUSED const runtime_shape_t &w_strides, NNCASE_UNUSED const runtime_shape_t &bias_strides, NNCASE_UNUSED const runtime_shape_t &out_strides,
    NNCASE_UNUSED const padding &padding_h, NNCASE_UNUSED const padding &padding_w,
    NNCASE_UNUSED int32_t groups, NNCASE_UNUSED int32_t stride_h, NNCASE_UNUSED int32_t stride_w,
    NNCASE_UNUSED int32_t dilation_h, NNCASE_UNUSED int32_t dilation_w, value_range<float> fused_activation, NNCASE_UNUSED kernel_context &context) noexcept
{
    const auto batch = in_shape[0], channels = w_shape[0], in_h = in_shape[2], in_w = in_shape[3];
    const auto out_h = kernels::detail::get_windowed_output_size(in_h, Filter_h, Stride_h, dilation_h, padding::zero());
    const auto out_w = kernels::detail::get_windowed_output_size(in_w, Filter_w, Stride_w, dilation_w, padding::zero());

    const size_t tail_step = in_strides[2] - (out_w * Stride_w);
    for (size_t b = 0; b < batch; b++) // batch
    {

#ifdef NNCASE_OPENMP
#pragma omp parallel for num_threads(GET_NUM_THREADS)
#endif
        for (size_t c = 0; c < channels; c++) // channel
        {
            std::array<float *, Parallel> outptr;
            std::array<const float *, compute_rsize<Parallel, Stride_h, Filter_h>()> r;
            std::array<const float *, Filter_h> k;
            std::array<float, Parallel> sum;

            float *out = output + out_strides[0] * b + out_strides[1] * c;
            std::fill_n(out, out_strides[2] ? out_h * out_strides[2] : (out_strides[3] ? out_w * out_strides[3] : 1), bias[c]);

            binding_ptr<Parallel>(outptr, out, out_strides[2]);
            binding_ptr<Parallel, Stride_h, Filter_h>(r, input + in_strides[0] * b + in_strides[1] * c, in_strides[2]);
            binding_ptr<Filter_h>(k, weights + w_strides[0] * c, w_strides[2]);
            conv2dChannel<Parallel, Filter_h, Filter_w, Stride_h, Stride_w>(out_h, out_w, sum, r, k, outptr, in_strides[2], out_strides[2], tail_step);
        }
    }
    // TODO use avx
    for (size_t _ = 0; _ < batch * channels * out_h * out_w; _++)
    {
        *(output + _) = kernels::detail::apply_activation(*(output + _), fused_activation);
    }
    return ok();
}

END_NS_NNCASE_KERNELS_CPU_OPT