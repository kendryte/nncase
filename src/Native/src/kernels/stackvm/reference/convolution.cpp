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
#include "ref_ops.h"
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;

template <typename T>
result<void> conv2d_impl(
    const T *input, const T *weights, const T *bias, T *output,
    std::span<const size_t> in_shape, std::span<const size_t> in_strides,
    std::span<const size_t> w_shape, std::span<const size_t> w_strides,
    std::span<const size_t> bias_strides, std::span<const size_t> out_strides,
    const padding &padding_h, const padding &padding_w, int32_t groups,
    int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
    value_range<float> fused_activation,
    NNCASE_UNUSED kernel_context &context) noexcept {
    const auto filter_h = (int32_t)w_shape[2];
    const auto filter_w = (int32_t)w_shape[3];
    const auto out_channels = w_shape[0];
    const auto out_h = kernels::detail::get_windowed_output_size(
        in_shape[2], filter_h, stride_h, dilation_h, padding_h);
    const auto out_w = kernels::detail::get_windowed_output_size(
        in_shape[3], filter_w, stride_w, dilation_w, padding_w);
    const auto g_ic = in_shape[1] / groups;
    const auto g_oc = out_channels / groups;

    dims_t in_index(4);
    dims_t w_index(4);
    dims_t bias_index(1);
    dims_t out_index(4);
    for (size_t batch = 0; batch < in_shape[0]; batch++) {
        in_index[0] = out_index[0] = batch;
        for (size_t og = 0; og < (size_t)groups; og++) {
            for (size_t oc = 0; oc < g_oc; oc++) {
                out_index[1] = w_index[0] = bias_index[0] = og * g_oc + oc;
                for (size_t oy = 0; oy < out_h; oy++) {
                    out_index[2] = oy;
                    for (size_t ox = 0; ox < out_w; ox++) {
                        out_index[3] = ox;
                        const int32_t in_y_origin =
                            (oy * stride_h) - padding_h.before;
                        const int32_t in_x_origin =
                            (ox * stride_w) - padding_w.before;
                        const int32_t filter_y_start = (int32_t)std::max(
                            0, (-in_y_origin + dilation_h - 1) / dilation_h);
                        const int32_t filter_y_end = (int32_t)std::min(
                            filter_h, ((int32_t)in_shape[2] - in_y_origin +
                                       dilation_h - 1) /
                                          dilation_h);
                        const int32_t filter_x_start = (int32_t)std::max(
                            0, (-in_x_origin + dilation_w - 1) / dilation_w);
                        const int32_t filter_x_end = (int32_t)std::min(
                            filter_w, ((int32_t)in_shape[3] - in_x_origin +
                                       dilation_w - 1) /
                                          dilation_w);
                        float value = bias[offset(bias_strides, bias_index)];

                        for (size_t ic = 0; ic < g_ic; ic++) {
                            in_index[1] = og * g_ic + ic;
                            w_index[1] = ic;
                            for (int32_t ky = filter_y_start; ky < filter_y_end;
                                 ky++) {
                                w_index[2] = ky;
                                for (int32_t kx = filter_x_start;
                                     kx < filter_x_end; kx++) {
                                    w_index[3] = kx;
                                    in_index[2] = in_y_origin + dilation_h * ky;
                                    in_index[3] = in_x_origin + dilation_w * kx;

                                    const T in_v =
                                        input[offset(in_strides, in_index)];
                                    const T w =
                                        weights[offset(w_strides, w_index)];

                                    value += in_v * w;
                                }
                            }
                        }

                        output[offset(out_strides, out_index)] =
                            kernels::detail::apply_activation(value,
                                                              fused_activation);
                    }
                }
            }
        }
    }

    return ok();
}

#define CONV2D_IMPL(type)                                                      \
    return conv2d_impl(                                                        \
        IN_CAST(type, input), IN_CAST(type, weights), IN_CAST(type, bias),     \
        OUT_CAST(type, output), in_shape, in_strides, w_shape, w_strides,      \
        bias_strides, out_strides, padding_h, padding_w, groups, stride_h,     \
        stride_w, dilation_h, dilation_w, fused_activation, context);

#define TYPE_SELECT_CONV(_typecode, _impl)                                     \
    switch (_typecode) {                                                       \
    case dt_float32:                                                           \
        _impl(float);                                                          \
    case dt_float16:                                                           \
        _impl(half);                                                           \
    case dt_bfloat16:                                                          \
        _impl(bfloat16);                                                       \
    case dt_float64:                                                           \
        _impl(double);                                                         \
    default:                                                                   \
        return err(std::errc::not_supported);                                  \
    }

result<void> nncase::kernels::stackvm::reference::conv2d(
    typecode_t typecode, const std::byte *input, const std::byte *weights,
    const std::byte *bias, std::byte *output, std::span<const size_t> in_shape,
    std::span<const size_t> in_strides, std::span<const size_t> w_shape,
    std::span<const size_t> w_strides, std::span<const size_t> bias_strides,
    std::span<const size_t> out_strides, const padding &padding_h,
    const padding &padding_w, int32_t groups, int32_t stride_h,
    int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
    value_range<float> fused_activation,
    NNCASE_UNUSED kernel_context &context) noexcept {
    TYPE_SELECT_CONV(typecode, CONV2D_IMPL);
}

template <typename T>
result<void> conv2d_transpose_impl(
    const T *input, T *output, const T *weights, const T *bias,
    std::span<const size_t> in_shape, int32_t groups,
    std::span<const size_t> out_shape, int32_t filter_h, int32_t filter_w,
    int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
    const padding &padding_h, const padding &padding_w,
    [[maybe_unused]] const value_range<float> &fused_activation) noexcept {
    auto output_size = runtime::compute_size(out_shape);
    std::fill(output, output + output_size, 0.f);
    const auto g_ic = in_shape[1] / groups;
    const auto g_oc = out_shape[1] / groups;

    for (size_t batch = 0; batch < in_shape[0]; batch++) {
        T *out_batch_p =
            output + (size_t)batch * out_shape[1] * out_shape[2] * out_shape[3];

        for (size_t g = 0; g < (size_t)groups; g++) {
            T *out_group_p =
                out_batch_p + (size_t)g * g_oc * out_shape[2] * out_shape[3];
            const T *w_group_p =
                weights + (size_t)g * g_oc * g_ic * filter_h * filter_w;

            for (size_t ic = 0; ic < g_ic; ic++) {
                for (size_t iy = 0; iy < in_shape[2]; iy++) {
                    for (size_t ix = 0; ix < in_shape[3]; ix++) {
                        const int32_t out_y_origin =
                            (iy * stride_h) - padding_h.before;
                        const int32_t out_x_origin =
                            (ix * stride_w) - padding_w.before;
                        const size_t filter_y_start = (size_t)std::max(
                            0, (-out_y_origin + dilation_h - 1) / dilation_h);
                        const size_t filter_y_end = (size_t)std::min(
                            filter_h, ((int32_t)out_shape[2] - out_y_origin +
                                       dilation_h - 1) /
                                          dilation_h);
                        const size_t filter_x_start = (size_t)std::max(
                            0, (-out_x_origin + dilation_w - 1) / dilation_w);
                        const size_t filter_x_end = (size_t)std::min(
                            filter_w, ((int32_t)out_shape[3] - out_x_origin +
                                       dilation_w - 1) /
                                          dilation_w);
                        const T in_v = *input++;

                        for (size_t oc = 0; oc < g_oc; oc++) {
                            T *out_c_p = out_group_p + (size_t)oc *
                                                           out_shape[2] *
                                                           out_shape[3];
                            const T *w_oc_p = w_group_p + (size_t)oc * g_ic *
                                                              filter_h *
                                                              filter_w;
                            const T *w_ic_p =
                                w_oc_p + (size_t)ic * filter_h * filter_w;

                            for (size_t ky = filter_y_start; ky < filter_y_end;
                                 ky++) {
                                for (size_t kx = filter_x_start;
                                     kx < filter_x_end; kx++) {
                                    const int32_t out_y =
                                        out_y_origin + dilation_h * ky;
                                    const int32_t out_x =
                                        out_x_origin + dilation_w * kx;

                                    const T w = w_ic_p[ky * filter_w + kx];

                                    out_c_p[out_y * out_shape[3] + out_x] +=
                                        in_v * w;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // bias
    auto hw = out_shape[2] * out_shape[3];
    for (size_t i = 0; i < output_size; i++)
        output[i] += bias[i / hw % out_shape[1]];

    //    if (fused_activation != value_range<float>::full())
    //    {
    //        for (size_t i = 0; i < output_size; i++)
    //            output[i] = detail::apply_activation(output[i],
    //            fused_activation);
    //    }
    return ok();
}

#define CONV2D_TRANSPOSE_IMPL(type)                                            \
    return conv2d_transpose_impl(                                              \
        IN_CAST(type, input), OUT_CAST(type, output), IN_CAST(type, weights),  \
        IN_CAST(type, bias), in_shape, groups, out_shape, filter_h, filter_w,  \
        stride_h, stride_w, dilation_h, dilation_w, padding_h, padding_w,      \
        fused_activation);

#define TYPE_SELECT_CONV_TRANSPOSE(_typecode, _impl)                           \
    switch (_typecode) {                                                       \
    case dt_float32:                                                           \
        _impl(float);                                                          \
    case dt_float16:                                                           \
        _impl(half);                                                           \
    case dt_bfloat16:                                                          \
        _impl(bfloat16);                                                       \
    case dt_float64:                                                           \
        _impl(double);                                                         \
    default:                                                                   \
        return err(std::errc::not_supported);                                  \
    }

result<void> nncase::kernels::stackvm::reference::conv2d_transpose(
    typecode_t typecode, const std::byte *input, std::byte *output,
    const std::byte *weights, const std::byte *bias,
    std::span<const size_t> in_shape, int32_t groups,
    std::span<const size_t> out_shape, int32_t filter_h, int32_t filter_w,
    int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
    const padding &padding_h, const padding &padding_w,
    [[maybe_unused]] const value_range<float> &fused_activation) noexcept {
    TYPE_SELECT_CONV_TRANSPOSE(typecode, CONV2D_TRANSPOSE_IMPL);
}
