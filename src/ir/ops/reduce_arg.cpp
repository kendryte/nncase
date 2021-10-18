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
#include <nncase/ir/ops/reduce_arg.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

reduce_arg::reduce_arg(reduce_arg_op_t op, datatype_t input_type, shape_t input_shape, datatype_t output_type, int32_t axis, bool keep_dims, bool select_last_index)
    : reduce_arg_op_(op), axis_(normalize_axis(input_shape, axis)), keep_dims_(keep_dims), select_last_index_(select_last_index)
{
    add_input("input", input_type, input_shape);
    axis_t axes { axis_ };
    add_output("output", output_type, get_reduced_shape(input_shape, axes, keep_dims));
}

bool reduce_arg::properties_equal(node &other) const
{
    auto &r = static_cast<reduce_arg &>(other);
    return reduce_arg_op() == r.reduce_arg_op() && axis() == r.axis() && keep_dims() == r.keep_dims() && select_last_index() == r.select_last_index();
}
