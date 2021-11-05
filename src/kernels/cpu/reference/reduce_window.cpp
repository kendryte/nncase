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
#include <nncase/kernels/cpu/reference/reduce_window.h>
#include <nncase/kernels/kernel_context.h>
#include <nncase/kernels/kernel_utils.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
struct identity_window
{
    float operator()(float src, NNCASE_UNUSED int32_t window) const noexcept
    {
        return src;
    }
};

template <class TBinaryOp, class TWindowOp>
result<void> reduce_window2d_impl(const float *input, float init_value, float *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const padding &padding_h, const padding &padding_w,
    int32_t filter_h, int32_t filter_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
    value_range<float> fused_activation, TBinaryOp &&binary_op, TWindowOp &&window_op, [[maybe_unused]] bool count_include_pad, NNCASE_UNUSED kernel_context &context) noexcept
{
    const auto out_h = kernels::detail::get_windowed_output_size(in_shape[2], filter_h, stride_h, dilation_h, padding_h);
    const auto out_w = kernels::detail::get_windowed_output_size(in_shape[3], filter_w, stride_w, dilation_w, padding_w);
    runtime_shape_t out_shape { in_shape[0], in_shape[1], out_h, out_w };

    for (size_t batch = 0; batch < in_shape[0]; batch++)
    {
        for (size_t oc = 0; oc < in_shape[1]; oc++)
        {
            for (size_t oy = 0; oy < out_h; oy++)
            {
                for (size_t ox = 0; ox < out_w; ox++)
                {
                    const int32_t in_y_origin = ((int32_t)oy * stride_h) - padding_h.before;
                    const int32_t in_x_origin = ((int32_t)ox * stride_w) - padding_w.before;
                    const size_t filter_y_start = (size_t)std::max(0, (-in_y_origin + dilation_h - 1) / dilation_h);
                    const size_t filter_y_end = (size_t)std::min(filter_h, ((int32_t)in_shape[2] - in_y_origin + dilation_h - 1) / dilation_h);
                    const size_t filter_x_start = (size_t)std::max(0, (-in_x_origin + dilation_w - 1) / dilation_w);
                    const size_t filter_x_end = (size_t)std::min(filter_w, ((int32_t)in_shape[3] - in_x_origin + dilation_w - 1) / dilation_w);
                    float value = init_value;
                    int32_t kernel_count = 0;

                    for (size_t ky = filter_y_start; ky < filter_y_end; ky++)
                    {
                        for (size_t kx = filter_x_start; kx < filter_x_end; kx++)
                        {
                            const size_t in_y = in_y_origin + dilation_h * ky;
                            const size_t in_x = in_x_origin + dilation_w * kx;

                            const float in_v = input[offset(in_strides, { batch, oc, in_y, in_x })];

                            value = binary_op(value, in_v);
                            kernel_count++;
                        }
                    }

                    if (count_include_pad)
                    {
                        for (int32_t i = 0; i < filter_h * filter_w - kernel_count; i++)
                            value = binary_op(value, 0.f);
                        kernel_count = filter_h * filter_w;
                    }

                    output[offset(out_strides, { batch, oc, oy, ox })] = kernels::detail::apply_activation(window_op(value, kernel_count), fused_activation);
                }
            }
        }
    }

    return ok();
}
}

#define REDUCE_WINDOW2D_IMPL(op, reducer, post_process) \
    case op:                                            \
        return reduce_window2d_impl(input, init_value, output, in_shape, in_strides, out_strides, padding_h, padding_w, filter_h, filter_w, stride_h, stride_w, dilation_h, dilation_w, fused_activation, reducer, post_process, count_include_pad, context)

#define REDUCE_WINDOW2D_IMPL_NO_POST(op, reducer) \
    case op:                                      \
        return reduce_window2d_impl(input, init_value, output, in_shape, in_strides, out_strides, padding_h, padding_w, filter_h, filter_w, stride_h, stride_w, dilation_h, dilation_w, fused_activation, reducer, identity_window(), count_include_pad, context)

result<void> reference::reduce_window2d(reduce_op_t op, const float *input, float init_value, float *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const padding &padding_h, const padding &padding_w,
    int32_t filter_h, int32_t filter_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation, bool count_include_pad,
    kernel_context &context) noexcept
{
    switch (op)
    {
        REDUCE_WINDOW2D_IMPL(reduce_mean, std::plus<float>(), [](float v, int32_t block_size) { return v / (float)block_size; });
        REDUCE_WINDOW2D_IMPL_NO_POST(reduce_min, [](float a, float b) { return std::min(a, b); });
        REDUCE_WINDOW2D_IMPL_NO_POST(reduce_max, [](float a, float b) { return std::max(a, b); });
        REDUCE_WINDOW2D_IMPL_NO_POST(reduce_sum, std::plus<float>());
    default:
        return err(std::errc::not_supported);
    }
}
