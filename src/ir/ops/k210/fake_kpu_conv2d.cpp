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
#include <ir/ops/k210/fake_kpu_conv2d.h>
#include <runtime/k210/k210_runtime_op_utility.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime::k210;

fake_kpu_conv2d::fake_kpu_conv2d(shape_t input_shape, bool is_depthwise, runtime::k210::kpu_filter_type_t filter_type, runtime::k210::kpu_pool_type_t pool_type, xt::xtensor<float, 4> weights, xt::xtensor<float, 1> bias, xt::svector<runtime::k210::piecewise_linear_segment> fused_activation)
    : weights_(std::move(weights)), bias_(std::move(bias)), is_depthwise_(is_depthwise), filter_type_(filter_type), pool_type_(pool_type), fused_activation_(fused_activation)
{
    add_input("input", dt_float32, input_shape);
    add_output("output", dt_float32,
        shape_t {
            input_shape[0],
            (size_t)output_channels(),
            (size_t)get_kpu_pool_output_size((int32_t)input_shape[2], pool_type_),
            (size_t)get_kpu_pool_output_size((int32_t)input_shape[3], pool_type_) });
}
