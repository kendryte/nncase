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
#include <nncase/ir/ops/reduce.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

reduce::reduce(reduce_op_t reduce_op, shape_t input_shape, axis_t axis, float init_value, bool keep_dims)
    : reduce_op_(reduce_op), axis_(normalize_reduce_axis(input_shape, axis)), init_value_(init_value), keep_dims_(keep_dims)
{
    add_input("input", dt_float32, input_shape);
    add_output("output", dt_float32, get_reduced_shape(input_shape, axis_, keep_dims_));
}

bool reduce::properties_equal(node &other) const
{
    auto &r = static_cast<reduce &>(other);
    return reduce_op() == r.reduce_op() && axis() == r.axis() && init_value() == r.init_value() && keep_dims() == r.keep_dims();
}
