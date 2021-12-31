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
#include <nncase/ir/ops/topk.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

topk::topk(datatype_t input_type, shape_t input_shape, int64_t k, int32_t axis, bool largest, bool sorted)
    : k_(k), axis_(normalize_axis(input_shape, axis)), largest_(largest), sorted_(sorted)
{
    shape_t output_shape = input_shape;
    output_shape[axis_] = k;

    add_input("input", input_type, input_shape);
    add_output("output_values", input_type, output_shape);
    add_output("output_indices", dt_int64, output_shape);
}

bool topk::properties_equal(node &other) const
{
    auto &r = static_cast<topk &>(other);
    return k() == r.k() && axis() == r.axis() && largest() == r.largest() && sorted() == r.sorted();
}
