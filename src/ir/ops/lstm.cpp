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

lstm::lstm(shape_t input_shape, shape_t w_shape, shape_t r_shape, shape_t b_shape, shape_t output_shape,
    shape_t output_h_shape, shape_t output_c_shape, bool has_static, lstm_direction direction, std::string framework)
    : has_static_(has_static), direction_(direction), framework_(framework)
{
    add_input("input", dt_float32, input_shape);
    add_input("w", dt_float32, w_shape);
    add_input("r", dt_float32, r_shape);
    add_input("b", dt_float32, b_shape);
    add_input("initial_h", dt_float32, output_h_shape);
    add_input("initial_h", dt_float32, output_c_shape);
    if (has_static)
        add_input("w_static", dt_float32, shape_t { w_shape[1], w_shape[2] });

    add_output("output", dt_float32, output_shape);
    add_output("output_h", dt_float32, output_h_shape);
    add_output("output_c", dt_float32, output_c_shape);
}

bool lstm::properties_equal(node &other) const
{
    auto &r = static_cast<lstm &>(other);
    return has_static() == r.has_static() && direction() == r.direction() && framework() == r.framework();
}
