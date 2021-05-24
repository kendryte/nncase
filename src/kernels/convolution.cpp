/* Copyright 2019-2020 Canaan Inc.
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
#include <nncase/kernels/convolution.h>
#include <nncase/kernels/cpu/optimized/convolution.h>
#include <nncase/kernels/cpu/reference/convolution.h>
#include <nncase/runtime/runtime_op_utility.h>
using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;

#define CONV_ARGS input, weights, bias, output,           \
                  in_shape, in_strides, w_shape,          \
                  w_strides, bias_strides, out_strides,   \
                  padding_h, padding_w, groups, stride_h, \
                  stride_w, dilation_h, dilation_w, fused_activation, context

result<void> kernels::conv2d(const float *input, const float *weights, const float *bias, float *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, const runtime_shape_t &w_strides,
    const runtime_shape_t &bias_strides, const runtime_shape_t &out_strides, const padding &padding_h, const padding &padding_w,
    int32_t groups, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation, kernel_context &context) noexcept
{
    const auto filter_h = w_shape[2];
    const auto filter_w = w_shape[3];
    if (groups != 1 and (size_t) groups == in_shape[1] and (size_t) groups == w_shape[0] and padding_h.before == 0 and padding_h.after == 0 and padding_w.before == 0 and padding_w.after == 0 and dilation_h == 1 and dilation_w == 1)
    {
        if (filter_h == 3 and filter_w == 3)
        {
            if (stride_h == 1 and stride_w == 1)
            {
                return cpu::optimized::conv2ddepthwise_NxM<8, 3, 3, 1, 1>(CONV_ARGS);
            }
            else if (stride_h == 2 and stride_w == 2)
            {
                return cpu::optimized::conv2ddepthwise_NxM<8, 3, 3, 2, 2>(CONV_ARGS);
            }
        }
        else if (filter_h == 3 and filter_w == 1)
        {
            if (stride_h == 1 and stride_w == 1)
            {
                return cpu::optimized::conv2ddepthwise_NxM<8, 3, 1, 1, 1>(CONV_ARGS);
            }
            else if (stride_h == 2 and stride_w == 2)
            {
                return cpu::optimized::conv2ddepthwise_NxM<8, 3, 1, 2, 2>(CONV_ARGS);
            }
        }
        else if (filter_h == 1 and filter_w == 3)
        {
            if (stride_h == 1 and stride_w == 1)
            {
                return cpu::optimized::conv2ddepthwise_NxM<8, 1, 3, 1, 1>(CONV_ARGS);
            }
            else if (stride_h == 2 and stride_w == 2)
            {
                return cpu::optimized::conv2ddepthwise_NxM<8, 1, 3, 2, 2>(CONV_ARGS);
            }
        }
        else if (filter_h == 5 and filter_w == 5)
        {
            if (stride_h == 1 and stride_w == 1)
            {
                return cpu::optimized::conv2ddepthwise_NxM<8, 5, 5, 1, 1>(CONV_ARGS);
            }
            else if (stride_h == 2 and stride_w == 2)
            {
                return cpu::optimized::conv2ddepthwise_NxM<8, 5, 5, 2, 2>(CONV_ARGS);
            }
        }
        else if (filter_h == 7 and filter_w == 7)
        {
            if (stride_h == 1 and stride_w == 1)
            {
                return cpu::optimized::conv2ddepthwise_NxM<8, 7, 7, 1, 1>(CONV_ARGS);
            }
            else if (stride_h == 2 and stride_w == 2)
            {
                return cpu::optimized::conv2ddepthwise_NxM<8, 7, 7, 2, 2>(CONV_ARGS);
            }
        }
    }

    if (groups == 1 and padding_h.before == 0 and padding_h.after == 0 and padding_w.before == 0 and padding_w.after == 0 and dilation_h == 1 and dilation_w == 1)
    {
        if (filter_h == 1 and filter_w == 1)
        {
            if (stride_h == 1 and stride_w == 1)
            {
                return cpu::optimized::conv2d_1x1_s1(CONV_ARGS);
            }
            else if (stride_h == 2 and stride_w == 2)
            {
                return cpu::optimized::conv2d_1x1_s2(CONV_ARGS);
            }
        }
        else if (filter_h == 1 and filter_w == 3)
        {
            if (stride_h == 1 and stride_w == 1)
            {
                return cpu::optimized::conv2d_NxM<8, 1, 3, 1, 1>(CONV_ARGS);
            }
            else if (stride_h == 2 and stride_w == 2)
            {
                return cpu::optimized::conv2d_NxM<8, 1, 3, 2, 2>(CONV_ARGS);
            }
        }
        else if (filter_h == 3 and filter_w == 1)
        {
            if (stride_h == 1 and stride_w == 1)
            {
                return cpu::optimized::conv2d_NxM<8, 3, 1, 1, 1>(CONV_ARGS);
            }
            else if (stride_h == 2 and stride_w == 2)
            {
                return cpu::optimized::conv2d_NxM<8, 3, 1, 2, 2>(CONV_ARGS);
            }
        }
        else if (filter_h == 1 and filter_w == 3)
        {
            if (stride_h == 1 and stride_w == 1)
            {
                return cpu::optimized::conv2d_NxM<8, 1, 3, 1, 1>(CONV_ARGS);
            }
            else if (stride_h == 2 and stride_w == 2)
            {
                return cpu::optimized::conv2d_NxM<8, 1, 3, 2, 2>(CONV_ARGS);
            }
        }
        else if (filter_h == 3 and filter_w == 3)
        {
            if (stride_h == 1 and stride_w == 1)
            {
                return cpu::optimized::conv2d_NxM<8, 3, 3, 1, 1>(CONV_ARGS);
            }
            else if (stride_h == 2 and stride_w == 2)
            {
                return cpu::optimized::conv2d_NxM<8, 3, 3, 2, 2>(CONV_ARGS);
            }
        }
        else if (filter_h == 5 and filter_w == 5)
        {
            if (stride_h == 1 and stride_w == 1)
            {
                return cpu::optimized::conv2d_NxM<8, 5, 5, 1, 1>(CONV_ARGS);
            }
            else if (stride_h == 2 and stride_w == 2)
            {
                return cpu::optimized::conv2d_NxM<8, 5, 5, 2, 2>(CONV_ARGS);
            }
        }
        else if (filter_h == 7 and filter_w == 7)
        {
            if (stride_h == 1 and stride_w == 1)
            {
                return cpu::optimized::conv2d_NxM<8, 7, 7, 1, 1>(CONV_ARGS);
            }
            else if (stride_h == 2 and stride_w == 2)
            {
                return cpu::optimized::conv2d_NxM<8, 7, 7, 2, 2>(CONV_ARGS);
            }
        }
    }
    // general conv
    return cpu::reference::conv2d(CONV_ARGS);
}
