/* Copyright 2019 Canaan Inc.
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
#include <ir/op_utils.h>
#include <ir/ops/conv2d_transpose.h>

using namespace nncase;
using namespace nncase::ir;

conv2d_transpose::conv2d_transpose(shape_t input_shape, shape_t output_shape, xt::xtensor<float, 4> weights, xt::xtensor<float, 1> bias, int32_t groups, padding padding_h, padding padding_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation)
    : weights_(std::move(weights)), bias_(std::move(bias)), groups_(groups), padding_h_(padding_h), padding_w_(padding_w), stride_h_(stride_h), stride_w_(stride_w), dilation_h_(dilation_h), dilation_w_(dilation_w), fused_activation_(fused_activation)
{
    if (get_windowed_output_size(output_shape[2] + padding_h_.sum(), weights_.shape()[2], stride_h_, dilation_h_, false) != input_shape[2]
        || get_windowed_output_size(output_shape[3] + padding_w_.sum(), weights_.shape()[3], stride_w_, dilation_w_, false) != input_shape[3])
        throw std::runtime_error("Invalid conv2d transpose shape");

    add_input("input", dt_float32, input_shape);
    add_output("output", dt_float32, output_shape);
}
