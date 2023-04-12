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
#include <nncase/kernels/reduce_window.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;

result<void> kernels::reduce_window2d(
    reduce_op_t op, const float *input, float init_value, float *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    const runtime_shape_t &out_strides, const padding &padding_h,
    const padding &padding_w, int32_t filter_h, int32_t filter_w,
    int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
    value_range<float> fused_activation, kernel_context &context) noexcept {
    return cpu::reference::reduce_window2d(
        op, input, init_value, output, in_shape, in_strides, out_strides,
        padding_h, padding_w, filter_h, filter_w, stride_h, stride_w,
        dilation_h, dilation_w, fused_activation, context);
}
