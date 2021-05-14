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

// NNCASE_API result<void> conv2d_1x1_s1(const float *input, const float *weights, const float *bias, float *output,
//     const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, const runtime_shape_t &w_strides,
//     const runtime_shape_t &bias_strides, const runtime_shape_t &out_strides, const padding &padding_h, const padding &padding_w,
//     int32_t groups, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation, kernel_context &context = default_kernel_context) noexcept;

// NNCASE_API result<void> conv2d_1x1_s2(const float *input, const float *weights, const float *bias, float *output,
//     const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, const runtime_shape_t &w_strides,
//     const runtime_shape_t &bias_strides, const runtime_shape_t &out_strides, const padding &padding_h, const padding &padding_w,
//     int32_t groups, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation, kernel_context &context = default_kernel_context) noexcept;

// NNCASE_API result<void> conv2d_3x3_s1(const float *input, const float *weights, const float *bias, float *output,
//     const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, const runtime_shape_t &w_strides,
//     const runtime_shape_t &bias_strides, const runtime_shape_t &out_strides, const padding &padding_h, const padding &padding_w,
//     int32_t groups, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation, kernel_context &context = default_kernel_context) noexcept;

// NNCASE_API result<void> conv2d_3x3_s2(const float *input, const float *weights, const float *bias, float *output,
//     const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, const runtime_shape_t &w_strides,
//     const runtime_shape_t &bias_strides, const runtime_shape_t &out_strides, const padding &padding_h, const padding &padding_w,
//     int32_t groups, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation, kernel_context &context = default_kernel_context) noexcept;

// NNCASE_API result<void> conv2d_5x5_s1(const float *input, const float *weights, const float *bias, float *output,
//     const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, const runtime_shape_t &w_strides,
//     const runtime_shape_t &bias_strides, const runtime_shape_t &out_strides, const padding &padding_h, const padding &padding_w,
//     int32_t groups, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation, kernel_context &context = default_kernel_context) noexcept;

// NNCASE_API result<void> conv2d_5x5_s2(const float *input, const float *weights, const float *bias, float *output,
//     const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, const runtime_shape_t &w_strides,
//     const runtime_shape_t &bias_strides, const runtime_shape_t &out_strides, const padding &padding_h, const padding &padding_w,
//     int32_t groups, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation, kernel_context &context = default_kernel_context) noexcept;

// NNCASE_API result<void> conv2d_7x7_s1(const float *input, const float *weights, const float *bias, float *output,
//     const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, const runtime_shape_t &w_strides,
//     const runtime_shape_t &bias_strides, const runtime_shape_t &out_strides, const padding &padding_h, const padding &padding_w,
//     int32_t groups, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation, kernel_context &context = default_kernel_context) noexcept;

size_t offset(const runtime_shape_t &strides, const runtime_shape_t &index)
{
    assert(strides.size() == index.size());
    return xt::element_offset<size_t>(strides, index.begin(), index.end());
}

namespace impl
{

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

template <typename T, size_t N, size_t... I>
void binding_value(std::array<T *, N> &a, std::array<T, N> &b, std::index_sequence<I...>)
{
    ((*a[I] += b[I]), ...);
}

template <typename Array, size_t... I>
void increase_n(Array &a, size_t step, std::index_sequence<I...>)
{
    ((a[I] += step), ...);
}

// template <size_t Start = 0, typename T, size_t N, size_t... I>
// auto binding_ptr(std::array<T, N> &a, T base, size_t step, std::index_sequence<I...>)
// {
//     ((a[Start + I] = base + step * I), ...);
// }

// template <size_t Filter, size_t Stride, typename T, size_t N, size_t... P>
// void binding_ptr(std::array<T, N> &a, T base, size_t step, std::index_sequence<P...>)
// {
//     (binding_ptr<P * std::min(Filter, Stride)>(a,
//          base + P * step * Stride, step, std::make_index_sequence<Filter> {}),
//         ...);
// }

} // namespace impl

// template <size_t Filter, size_t Start = 0, typename T, size_t N>
// void binding_ptr(std::array<T, N> &a, T base, size_t step)
// {
//     impl::binding_ptr<Start>(a, base, step, std::make_index_sequence<Filter> {});
// }

// template <size_t Parallel, size_t Stride, size_t Filter, typename T, size_t N>
// void binding_ptr(std::array<T, N> &a, T base, size_t step)
// {
//     impl::binding_ptr<Filter, Stride>(a, base, step, std::make_index_sequence<Parallel> {});
// }

template <size_t Parallel, size_t P, size_t Stride_h, size_t Filter_h,
    size_t Filter_w, typename T, size_t N, size_t X>
void convNxM(std::array<T, X> &sum, std::array<const T *, N> &r,
    std::array<const T *, Filter_h> &k)
{
    impl::convNxM<P * std::min(Stride_h, Filter_h), Filter_h, Filter_w>(sum[P], r, k);
    if constexpr (P < Parallel - 1)
    {
        convNxM<Parallel, P + 1, Stride_h, Filter_h, Filter_w, T, N, X>(sum, r, k);
    }
}

template <size_t Filter, typename T, size_t N>
void binding_ptr(std::array<T, N> &a, T base, size_t step, size_t start = 0)
{
    for (size_t i = 0; i < Filter; i++)
    {
        a[start + i] = base + step * i;
    }
}

template <size_t Parallel, size_t Stride, size_t Filter, typename T, size_t N>
void binding_ptr(std::array<T, N> &a, T base, size_t step)
{
    for (size_t p = 0; p < Parallel; p++)
    {
        binding_ptr<Filter>(a, base + p * Stride * step, step,
            p * std::min(Filter, Stride));
    }
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
    impl::binding_value(output, value, std::make_index_sequence<Parallel> {});
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
    impl::increase_n(a, step, std::make_index_sequence<N> {});
}

constexpr size_t compute_rsize(size_t Parallel, size_t Stride, size_t Filter)
{
    return Filter + std::min(Stride, Filter) * (Parallel - 1);
}

NNCASE_API template <size_t Parallel, size_t Filter_h, size_t Filter_w, size_t Stride_h, size_t Stride_w>
result<void> conv2d_3x3_s1(const float *input, const float *weights, const float *bias, float *output,
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
    std::array<const float *, compute_rsize(Parallel, Stride_h, Filter_h)> r;
    std::array<const float *, Filter_h> k;
    std::array<float, Parallel> sum;
    const size_t tail_size = in_strides[2] - (out_w * Stride_w);
    for (size_t b = 0; b < batch; b++) // batch
    {
        in_index[0] = out_index[0] = b;
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

                size_t i = 0;
                for (; i + (Parallel - 1) < out_h; i += Parallel)
                {
                    for (size_t remain = 0; remain < out_w; remain++)
                    {
                        std::fill_n(sum.begin(), Parallel, 0.);
                        convNxM<Parallel, 0, Stride_h, Filter_h, Filter_w>(sum, r, k);
                        binding_value<Parallel>(outptr, sum);
                        increase_n<compute_rsize(Parallel, Stride_h, Filter_h)>(r, Stride_w);
                        increase_n<Parallel>(outptr, 1);
                    }
                    increase_n<compute_rsize(Parallel, Stride_h, Filter_h)>(r,
                        in_strides[2] * (Stride_h - 1) + in_strides[2] * (Parallel - 1) + tail_size);
                    // FIXME 这里要判断一下 为0时就不展开了
                    increase_n<Parallel>(outptr, out_strides[2] * (Parallel - 1));
                }

                constexpr size_t Local_Parallel = 1;
                for (; i + (Local_Parallel - 1) < out_h; i += Local_Parallel)
                {
                    for (size_t remain = 0; remain < out_w; remain++)
                    {
                        std::fill_n(sum.begin(), Local_Parallel, 0.);
                        // convNxN(sum[0], r[0], k[0], r[1], k[1], r[2], k[2]);
                        convNxM<Local_Parallel, 0, Stride_h, Filter_h, Filter_w>(sum, r, k);
                        binding_value<Local_Parallel>(outptr, sum);
                        increase_n<compute_rsize(Local_Parallel, Stride_h, Filter_h)>(r, Stride_w);
                        increase_n<Local_Parallel>(outptr, 1);
                    }
                    increase_n<compute_rsize(Local_Parallel, Stride_h, Filter_h)>(r, in_strides[2] * (Stride_h - 1) + (in_strides[2] * (Local_Parallel - 1)) + tail_size);
                    // FIXME 这里要判断一下 为0时就不展开了
                    if constexpr (Local_Parallel - 1 > 0)
                    {
                        // FIXME 这里可能得考虑一下内存是否连续的问题
                        increase_n<Local_Parallel>(outptr, out_strides[2] * (Local_Parallel - 1));
                    }
                }
            }
        }
    }
    for (size_t _ = 0; _ < batch * out_channels * out_h * out_w; _++)
    {
        *(output + _) = kernels::detail::apply_activation(*(output + _), fused_activation);
    }
    return ok();
}

// NNCASE_API template <size_t Parallel, size_t Filter_h, size_t Filter_w, size_t Stride_h, size_t Stride_w>
// result<void> conv2d_7x7_s1(const float *input, const float *weights, const float *bias, float *output,
//     const runtime_shape_t &in_shape, NNCASE_UNUSED const runtime_shape_t &in_strides, NNCASE_UNUSED const runtime_shape_t &w_shape,
//     NNCASE_UNUSED const runtime_shape_t &w_strides, NNCASE_UNUSED const runtime_shape_t &bias_strides, NNCASE_UNUSED const runtime_shape_t &out_strides,
//     NNCASE_UNUSED const padding &padding_h, NNCASE_UNUSED const padding &padding_w,
//     NNCASE_UNUSED int32_t groups, NNCASE_UNUSED int32_t stride_h, NNCASE_UNUSED int32_t stride_w,
//     NNCASE_UNUSED int32_t dilation_h, NNCASE_UNUSED int32_t dilation_w, value_range<float> fused_activation, NNCASE_UNUSED kernel_context &context) noexcept
// {
//     const auto batch = in_shape[0], out_channels = w_shape[0], in_channels = w_shape[1], in_h = in_shape[2], in_w = in_shape[3];
//     const auto out_h = kernels::detail::get_windowed_output_size(in_h, Filter_h, stride_h, dilation_h, padding::zero());
//     const auto out_w = kernels::detail::get_windowed_output_size(in_w, Filter_w, stride_w, dilation_w, padding::zero());
//     runtime_shape_t in_index(4, 0), out_index(4, 0), w_index(4, 0);
//     const size_t tail_size = in_strides[2] - (out_w * stride_w);
//     constexpr size_t R_size = Filter_h + std::min(Stride_h, Filter_h) * (Parallel - 1);
//     // constexpr size_t R_size=8;
//     std::array<float *, Parallel> outptr;
//     // std::cout << "\nR_size : " << R_size << std::endl;
//     std::array<const float *, R_size> r;
//     std::array<const float *, Filter_h> k;
//     std::array<float, Parallel> sum;
//     // FIXME 这里的一些size一开始可以申请8为静态，后面递归起来还是要动态计算，可能需要写个宏
//     for (size_t b = 0; b < batch; b++) // batch
//     {
//         in_index[0] = out_index[0] = b;
//         // #pragma omp Parallel for num_threads(opt.num_threads)
//         for (size_t oc = 0; oc < out_channels; oc++) // out channel
//         {
//             out_index[1] = w_index[0] = oc;
//             float *out = output + offset(out_strides, out_index);

//             std::fill(out, out + out_h * out_w, bias[oc]);

//             for (size_t ic = 0; ic < in_channels; ic++) // in channel
//             {
//                 in_index[1] = w_index[1] = ic;

//                 binding_ptr<Parallel>(outptr, out, out_strides[2]);
//                 binding_ptr<Parallel, Stride_h, Filter_h>(r, input + offset(in_strides, in_index), in_strides[2]);
//                 binding_ptr<Filter_h>(k, weights + offset(w_strides, w_index), w_strides[2]);

//                 size_t i = 0;
//                 // for (; i + (Parallel - 1) < out_h; i += Parallel)
//                 // {
//                 //     for (size_t remain = 0; remain < out_w; remain++)
//                 //     {
//                 //         std::fill_n(sum, parallel, 0.);
//                 //         // FIXME 使用静态推导进行展开
//                 //         convNxN(sum[0], r[0], k[0], r[1], k[1], r[2], k[2],
//                 //             r[3], k[3], r[4], k[4], r[5], k[5],
//                 //             r[6], k[6]);
//                 //         convNxN(sum[1], r[1], k[0], r[2], k[1], r[3], k[2],
//                 //             r[4], k[3], r[5], k[4], r[6], k[5],
//                 //             r[7], k[6]);
//                 //         binding_value(outptr, sum, parallel);
//                 //         increase_n(r, k_size + 1, stride_h);
//                 //         increase_n(outptr, parallel);
//                 //     }
//                 //     // FIXME 在行不连续的情况下跳跃的步长需要更加细致的计算。
//                 //     increase_n(r, k_size + 1, in_strides[2] * stride_h + tail_size);
//                 //     increase_n(outptr, parallel, out_strides[2]);
//                 // }
//                 constexpr size_t Local_Parallel = 1;
//                 for (; i + (Local_Parallel - 1) < out_h; i += Local_Parallel)
//                 {
//                     for (size_t remain = 0; remain < out_w; remain++)
//                     {
//                         std::fill_n(sum.begin(), 1, 0.);
//                         convNxN(sum[0], r[0], k[0], r[1], k[1], r[2], k[2],
//                             r[3], k[3], r[4], k[4], r[5], k[5],
//                             r[6], k[6]);
//                         binding_value<Local_Parallel>(outptr, sum);
//                         increase_n<Filter_h + std::min(Stride_h, Filter_h) * (Local_Parallel - 1)>(r);
//                         increase_n<Local_Parallel>(outptr);
//                     }
//                     increase_n<Filter_h + std::min(Stride_h, Filter_h) * (Local_Parallel - 1)>(r, in_strides[2] * Stride_h + tail_size);
//                     // FIXME 这里要判断一下 为0时就不展开了
//                     increase_n<Local_Parallel>(outptr, out_strides[2] * (Local_Parallel - 1));
//                 }
//             }
//         }
//     }
//     // FIXME 改进遍历的机制
//     for (size_t _ = 0; _ < batch * out_channels * out_h * out_w; _++)
//     {
//         *(output + _) = kernels::detail::apply_activation(*(output + _), fused_activation);
//     }
//     return ok();
// }

// NNCASE_API template <size_t Parallel, size_t Filter_h, size_t Filter_w, size_t Stride_h, size_t Stride_w>
// result<void> conv2d_7x7_s2(const float *input, const float *weights, const float *bias, float *output,
//     const runtime_shape_t &in_shape, NNCASE_UNUSED const runtime_shape_t &in_strides, NNCASE_UNUSED const runtime_shape_t &w_shape,
//     NNCASE_UNUSED const runtime_shape_t &w_strides, NNCASE_UNUSED const runtime_shape_t &bias_strides, NNCASE_UNUSED const runtime_shape_t &out_strides,
//     NNCASE_UNUSED const padding &padding_h, NNCASE_UNUSED const padding &padding_w,
//     NNCASE_UNUSED int32_t groups, NNCASE_UNUSED int32_t stride_h, NNCASE_UNUSED int32_t stride_w,
//     NNCASE_UNUSED int32_t dilation_h, NNCASE_UNUSED int32_t dilation_w, value_range<float> fused_activation, NNCASE_UNUSED kernel_context &context) noexcept
// {
//     const auto batch = in_shape[0], out_channels = w_shape[0], in_channels = w_shape[1], in_h = in_shape[2], in_w = in_shape[3];
//     const auto out_h = kernels::detail::get_windowed_output_size(in_h, Filter_h, stride_h, dilation_h, padding::zero());
//     const auto out_w = kernels::detail::get_windowed_output_size(in_w, Filter_w, stride_w, dilation_w, padding::zero());
//     runtime_shape_t in_index(4, 0), out_index(4, 0), w_index(4, 0);
//     const size_t tail_size = in_strides[2] - (out_w * stride_w);
//     constexpr size_t R_size = Filter_h + std::min(Stride_h, Filter_h) * (Parallel - 1);
//     std::array<float *, Parallel> outptr;
//     std::array<const float *, R_size> r;
//     std::array<const float *, Filter_h> k;
//     std::array<float, Parallel> sum;
//     // FIXME 这里的一些size一开始可以申请8为静态，后面递归起来还是要动态计算，可能需要写个宏
//     for (size_t b = 0; b < batch; b++) // batch
//     {
//         in_index[0] = out_index[0] = b;
//         // #pragma omp Parallel for num_threads(opt.num_threads)
//         for (size_t oc = 0; oc < out_channels; oc++) // out channel
//         {
//             out_index[1] = w_index[0] = oc;
//             float *out = output + offset(out_strides, out_index);

//             std::fill(out, out + out_h * out_w, bias[oc]);

//             for (size_t ic = 0; ic < in_channels; ic++) // in channel
//             {
//                 in_index[1] = w_index[1] = ic;

//                 binding_ptr<Parallel>(outptr, out, out_strides[2]);
//                 binding_ptr<Parallel, Stride_h, Filter_h>(r, input + offset(in_strides, in_index), in_strides[2]);
//                 binding_ptr<Filter_h>(k, weights + offset(w_strides, w_index), w_strides[2]);

//                 size_t i = 0;
//                 // for (; i + (Parallel - 1) < out_h; i += Parallel)
//                 // {
//                 //     for (size_t remain = 0; remain < out_w; remain++)
//                 //     {
//                 //         std::fill_n(sum, parallel, 0.);
//                 //         // FIXME 使用静态推导进行展开
//                 //         convNxN(sum[0], r[0], k[0], r[1], k[1], r[2], k[2],
//                 //             r[3], k[3], r[4], k[4], r[5], k[5],
//                 //             r[6], k[6]);
//                 //         convNxN(sum[1], r[1], k[0], r[2], k[1], r[3], k[2],
//                 //             r[4], k[3], r[5], k[4], r[6], k[5],
//                 //             r[7], k[6]);
//                 //         binding_value(outptr, sum, parallel);
//                 //         increase_n(r, k_size + 1, stride_h);
//                 //         increase_n(outptr, parallel);
//                 //     }
//                 //     // FIXME 在行不连续的情况下跳跃的步长需要更加细致的计算。
//                 //     increase_n(r, k_size + 1, in_strides[2] * stride_h + tail_size);
//                 //     increase_n(outptr, parallel, out_strides[2]);
//                 // }
//                 constexpr size_t Local_Parallel = 1;
//                 for (; i + (Local_Parallel - 1) < out_h; i += Local_Parallel)
//                 {
//                     for (size_t remain = 0; remain < out_w; remain++)
//                     {
//                         std::fill_n(sum.begin(), 1, 0.);
//                         convNxN(sum[0], r[0], k[0], r[1], k[1], r[2], k[2],
//                             r[3], k[3], r[4], k[4], r[5], k[5],
//                             r[6], k[6]);
//                         binding_value<Local_Parallel>(outptr, sum);
//                         increase_n<Filter_h + std::min(Stride_h, Filter_h) * (Local_Parallel - 1)>(r);
//                         increase_n<Local_Parallel>(outptr);
//                     }
//                     increase_n<Filter_h + std::min(Stride_h, Filter_h) * (Local_Parallel - 1)>(r, in_strides[2] * Stride_h + tail_size);
//                     // FIXME 这里要判断一下 为0时就不展开了
//                     increase_n<Local_Parallel>(outptr, out_strides[2] * (Local_Parallel - 1));
//                 }
//             }
//         }
//     }
//     // FIXME 改进遍历的机制
//     for (size_t _ = 0; _ < batch * out_channels * out_h * out_w; _++)
//     {
//         *(output + _) = kernels::detail::apply_activation(*(output + _), fused_activation);
//     }
//     return ok();
// }

END_NS_NNCASE_KERNELS_CPU_OPT