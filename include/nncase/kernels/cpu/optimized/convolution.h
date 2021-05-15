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
#include <utility>

BEGIN_NS_NNCASE_KERNELS_CPU_OPT

template <size_t Parallel, size_t Stride, size_t Filter>
constexpr size_t compute_rsize()
{
    return Filter + std::min(Stride, Filter) * (Parallel - 1);
}

template <typename T, size_t... W>
void conv1xM(T &sum, const T *r, const T *k, std::index_sequence<W...>)
{
    ((sum += r[W] * k[W]), ...);
}

template <size_t R, size_t Filter_h, size_t Filter_w, typename T, size_t N,
    size_t... H>
void convNxM(T &sum, std::array<const T *, N> &r, std::array<const T *, Filter_h> &k,
    std::index_sequence<H...>)
{
    (conv1xM(sum, r[R + H], k[H], std::make_index_sequence<Filter_w> {}), ...);
}

template <size_t R, size_t Filter_h, size_t Filter_w, typename T, size_t N>
void convNxM(T &sum, std::array<const T *, N> &r, std::array<const T *, Filter_h> &k)
{
    convNxM<R, Filter_h, Filter_w>(sum, r, k,
        std::make_index_sequence<Filter_h> {});
}

template <size_t Parallel, size_t P, size_t Stride_h, size_t Filter_h,
    size_t Filter_w, typename T, size_t N, size_t X>
void convNxM(std::array<T, X> &sum, std::array<const T *, N> &r,
    std::array<const T *, Filter_h> &k)
{
    convNxM<P * std::min(Stride_h, Filter_h), Filter_h, Filter_w>(sum[P], r, k);
    if constexpr (P < Parallel - 1)
    {
        convNxM<Parallel, P + 1, Stride_h, Filter_h, Filter_w, T, N, X>(sum, r, k);
    }
}

template <typename T, size_t N, size_t... I>
void binding_ptr(std::array<T, N> &a, T base, size_t step, size_t start, std::index_sequence<I...>)
{
    ((a[start + I] = base + step * I), ...);
}

template <size_t Filter, typename T, size_t N>
void binding_ptr(std::array<T, N> &a, T base, size_t step, size_t start = 0)
{
    binding_ptr(a, base, step, start, std::make_index_sequence<Filter> {});
}

template <size_t Stride, size_t Filter, typename T, size_t N, size_t... P>
void binding_ptr(std::array<T, N> &a, T base, size_t step, std::index_sequence<P...>)
{
    (binding_ptr<Filter>(a, base + P * Stride * step, step,
         P * std::min(Filter, Stride)),
        ...);
}

template <size_t Parallel, size_t Stride, size_t Filter, typename T, size_t N>
void binding_ptr(std::array<T, N> &a, T base, size_t step)
{
    binding_ptr<Stride, Filter>(a, base, step, std::make_index_sequence<Parallel>());
}

template <typename T, size_t N, size_t... I>
void binding_value(std::array<T *, N> &a, std::array<T, N> &b, std::index_sequence<I...>)
{
    ((*a[I] += b[I]), ...);
}

/**
 * @brief binding value for output, *out[i]+=in[i], i ≤ Parallel ≤ N
 * 
 * @tparam Parallel current for loop times
 * @tparam Array 
 * @tparam T 
 * @tparam N
 * @param output
 * @param value
 */
template <size_t Parallel, typename T, size_t N>
void binding_value(std::array<T *, N> &output, std::array<T, N> &value)
{
    binding_value(output, value, std::make_index_sequence<Parallel> {});
}

template <typename Array, size_t... I>
void increase_n(Array &a, size_t step, std::index_sequence<I...>)
{
    ((a[I] += step), ...);
}

/**
 * @brief array self increase , a[i]+=step, i ≤ N ≤ A.size()
 * 
 * @tparam N 
 * @param a 
 * @param step 
 */
template <size_t N, typename Array>
void increase_n(Array &a, size_t step = 1)
{
    increase_n(a, step, std::make_index_sequence<N> {});
}

template <size_t LocalParallel, size_t Filter_h, size_t Filter_w, size_t Stride_h, size_t Stride_w, typename T, size_t R, size_t Parallel>
void conv2dChannel(size_t &i, size_t out_h, size_t out_w, std::array<T, Parallel> &sum, std::array<const T *, R> &r, std::array<const T *, Filter_h> k,
    std::array<T *, Parallel> outptr, size_t in_w_step, size_t out_w_step, size_t tail_step, T bias)
{
    for (; i + (LocalParallel - 1) < out_h; i += LocalParallel)
    {
        for (size_t remain = 0; remain < out_w; remain++)
        {
            std::fill_n(sum.begin(), LocalParallel, bias);
            convNxM<LocalParallel, 0, Stride_h, Filter_h, Filter_w>(sum, r, k);
            binding_value<LocalParallel>(outptr, sum);
            increase_n<compute_rsize<LocalParallel, Stride_h, Filter_h>()>(r, Stride_w);
            increase_n<LocalParallel>(outptr, 1);
        }
        increase_n<compute_rsize<LocalParallel, Stride_h, Filter_h>()>(r,
            (Stride_h * LocalParallel - 1) * in_w_step + tail_step);
        if constexpr (LocalParallel > 1)
        {
            increase_n<LocalParallel>(outptr, out_w_step * (LocalParallel - 1));
        }
    }
    if constexpr (LocalParallel > 2)
    {
        conv2dChannel<LocalParallel / 2, Filter_h, Filter_w, Stride_h, Stride_w>(i, out_h, out_w, sum, r, k, outptr, in_w_step, out_w_step, tail_step, bias);
    }
}

template <size_t LocalParallel, size_t Filter_h, size_t Filter_w, size_t Stride_h, size_t Stride_w, typename T, size_t R, size_t Parallel>
void conv2dChannel(size_t out_h, size_t out_w, std::array<T, Parallel> &sum, std::array<const T *, R> &r, std::array<const T *, Filter_h> k,
    std::array<T *, Parallel> outptr, size_t in_w_step, size_t out_w_step, size_t tail_step, T bias)
{
    size_t i = 0;
    conv2dChannel<LocalParallel, Filter_h, Filter_w, Stride_h, Stride_w>(i, out_h, out_w, sum, r, k, outptr, in_w_step, out_w_step, tail_step, bias);
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
    runtime_shape_t in_index(4, 0), out_index(4, 0), w_index(4, 0);
    std::array<float *, Parallel> outptr;
    std::array<const float *, compute_rsize<Parallel, Stride_h, Filter_h>()> r;
    std::array<const float *, Filter_h> k;
    std::array<float, Parallel> sum;
    const size_t tail_step = in_strides[2] - (out_w * Stride_w);
    constexpr float default_bias = 0;
    for (size_t b = 0; b < batch; b++) // batch
    {
        in_index[0] = out_index[0] = b;
        // TODO add omp parallel
        // #pragma omp parallel for num_threads(opt.num_threads)
        for (size_t oc = 0; oc < out_channels; oc++) // out channel
        {
            out_index[1] = w_index[0] = oc;
            float *out = output + offset(out_strides, out_index);

            std::fill(out, out + out_h * out_w, bias[oc]);

            for (size_t ic = 0; ic < in_channels; ic++) // in channel
            {
                in_index[1] = w_index[1] = ic;
                binding_ptr<Parallel>(outptr, out, out_strides[2]);
                binding_ptr<Parallel, Stride_h, Filter_h>(r, input + offset(in_strides, in_index), in_strides[2]);
                binding_ptr<Filter_h>(k, weights + offset(w_strides, w_index), w_strides[2]);
                conv2dChannel<Parallel, Filter_h, Filter_w, Stride_h, Stride_w>(out_h, out_w, sum, r, k, outptr, in_strides[2], out_strides[2], tail_step, default_bias);
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
    runtime_shape_t in_index(4, 0), out_index(4, 0), w_index(4, 0);
    std::array<float *, Parallel> outptr;
    std::array<const float *, compute_rsize<Parallel, Stride_h, Filter_h>()> r;
    std::array<const float *, Filter_h> k;
    std::array<float, Parallel> sum;
    const size_t tail_step = in_strides[2] - (out_w * Stride_w);
    for (size_t b = 0; b < batch; b++) // batch
    {
        in_index[0] = out_index[0] = b;
        // TODO add omp parallel
        // #pragma omp parallel for num_threads(opt.num_threads)
        for (size_t c = 0; c < channels; c++) // channel
        {
            in_index[1] = out_index[1] = w_index[0] = c;
            float *out = output + offset(out_strides, out_index);
            binding_ptr<Parallel>(outptr, out, out_strides[2]);
            binding_ptr<Parallel, Stride_h, Filter_h>(r, input + offset(in_strides, in_index), in_strides[2]);
            binding_ptr<Filter_h>(k, weights + offset(w_strides, w_index), w_strides[2]);
            conv2dChannel<Parallel, Filter_h, Filter_w, Stride_h, Stride_w>(out_h, out_w, sum, r, k, outptr, in_strides[2], out_strides[2], tail_step, bias[c]);
        }
    }
    for (size_t _ = 0; _ < batch * channels * out_h * out_w; _++)
    {
        *(output + _) = kernels::detail::apply_activation(*(output + _), fused_activation);
    }
    return ok();
}

END_NS_NNCASE_KERNELS_CPU_OPT