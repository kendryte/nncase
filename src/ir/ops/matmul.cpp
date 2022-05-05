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
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/matmul.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

matmul::matmul(shape_t input_a_shape, shape_t input_b_shape, value_range<float> fused_activation)
    : fused_activation_(fused_activation)
{
    add_input("input_a", dt_float32, input_a_shape);
    add_input("input_b", dt_float32, input_b_shape);
    add_input("bias", dt_float32, shape_t { input_b_shape.back() });
    add_output("output", dt_float32, get_matmul_output_shape(input_a_shape, input_b_shape));
}

bool matmul::properties_equal(node &other) const
{
    auto &r = static_cast<matmul &>(other);
    return fused_activation() == r.fused_activation();
}
