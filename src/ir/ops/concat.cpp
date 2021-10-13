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
#include <nncase/ir/ops/concat.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

concat::concat(datatype_t type, std::span<shape_t> input_shapes, int32_t axis)
    : axis_(normalize_axis(input_shapes[0], axis))
{
    if (input_shapes.empty())
        throw std::invalid_argument("there must be at least one input");

    for (size_t i = 0; i < input_shapes.size(); i++)
    {
        add_input("input_" + std::to_string(i), type, input_shapes[i]);
        concat_dims_.emplace_back(input_shapes[i][axis_]);
    }

    add_output("output", type, get_concated_shape(input_shapes, size_t(axis_)));
}

bool concat::properties_equal(node &other) const
{
    auto &r = static_cast<concat &>(other);
    return axis() == r.axis();
}
