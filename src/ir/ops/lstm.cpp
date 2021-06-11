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
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/lstm.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

lstm::lstm(shape_t input_a_shape, shape_t input_b_shape, int32_t num_output)
    : num_output_(num_output)
{
    add_input("input_a", dt_float32, input_a_shape);
    add_input("input_b", dt_float32, input_b_shape);
    add_output("output", dt_float32, shape_t { 128, 128 });
}

bool lstm::properties_equal(node &other) const
{
    auto &r = static_cast<lstm &>(other);
    return num_output() == r.num_output();
}
