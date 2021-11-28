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

lstm::lstm(shape_t input_shape, shape_t w_xc_shape, shape_t b_xc_shape, shape_t w_rc_shape, shape_t b_rc_shape, shape_t initial_h_shape, shape_t initial_c_shape, int32_t num_output, bool has_static, std::string framework)
    : num_output_(num_output), has_static_(has_static), framework_(framework)
{
    add_input("input", dt_float32, input_shape);
    add_input("w_xc", dt_float32, w_xc_shape);
    add_input("b_xc", dt_float32, b_xc_shape);
    add_input("w_rc", dt_float32, w_rc_shape);
    add_input("b_rc", dt_float32, b_rc_shape);
    add_input("initial_h", dt_float32, initial_h_shape);
    add_input("initial_h", dt_float32, initial_c_shape);
    if (has_static)
        add_input("w_static", dt_float32, shape_t { w_xc_shape[1], w_xc_shape[2] });

    add_output("output", dt_float32, shape_t { input_shape[0], input_shape[1], (size_t)num_output });
}

bool lstm::properties_equal(node &other) const
{
    auto &r = static_cast<lstm &>(other);
    return num_output() == r.num_output() && has_static() == r.has_static();
}
