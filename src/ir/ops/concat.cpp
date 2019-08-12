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
#include <ir/ops/concat.h>
#include <ir/op_utils.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

concat::concat(datatype_t type, xtl::span<shape_t> input_shapes, int32_t axis)
    : axis_(axis)
{
    if (input_shapes.empty())
        throw std::invalid_argument("there must be at least one input");

    for (size_t i = 0; i < input_shapes.size(); i++)
    {
        add_input("input_" + std::to_string(i), type, input_shapes[i]);
        concat_dims_.emplace_back(input_shapes[i][axis]);
    }

    add_output("output", type, get_concated_shape(input_shapes, axis));
}
