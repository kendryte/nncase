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
#include <nncase/ir/ops/split.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

split::split(datatype_t type, shape_t input_shape, std::vector<shape_t> outputs_shape, std::vector<size_t> indices_or_sections, int32_t axis, bool is_indices)
    : indices_or_sections_(indices_or_sections)
    , axis_(axis)
    , is_indices_(is_indices)
{
    add_input("input", type, input_shape);
    for (size_t i = 0; i < outputs_shape.size(); i++)
        add_output("output", type, outputs_shape[i]);
}

bool split::properties_equal(node &other) const
{
    auto &r = static_cast<split &>(other);
    return indices_or_sections() == r.indices_or_sections() && axis() == r.axis() && is_indices() == r.is_indices();
}
