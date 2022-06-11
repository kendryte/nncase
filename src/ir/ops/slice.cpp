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
#include <nncase/ir/ops/slice.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

slice::slice(datatype_t type, shape_t input_shape, axis_t begin, axis_t end, axis_t strides, int32_t begin_mask, int32_t end_mask, int32_t ellipsis_mask, int32_t new_axis_mask, int32_t shrink_axis_mask)
    : begin_(normalize_strided_slice_begin(input_shape, begin, strides, begin_mask))
    , end_(normalize_strided_slice_end(input_shape, begin_, end, strides, end_mask, shrink_axis_mask))
    , strides_(std::move(strides))
    , begin_mask_(0)
    , end_mask_(0)
    , ellipsis_mask_(ellipsis_mask)
    , new_axis_mask_(new_axis_mask)
    , shrink_axis_mask_(shrink_axis_mask)
{
    add_input("input", type, input_shape);
    add_output("output", type, get_strided_slice_output_shape(begin, end_, strides_, ellipsis_mask_, new_axis_mask_, shrink_axis_mask_));

    for (size_t i = 0; i < input_shape.size(); i++)
    {
        if (begin_[i] >= input_shape[i])
        {
            throw std::invalid_argument("Invalid slice begin " + std::to_string(begin_[i])
                + ", while dim is " + std::to_string(input_shape[i]));
        }

        if (end_[i] > static_cast<int32_t>(input_shape[i]))
        {
            throw std::invalid_argument("Invalid slice end " + std::to_string(begin_[i])
                + ", while dim is " + std::to_string(input_shape[i]));
        }
    }
}

slice::slice(datatype_t type, shape_t input_shape, axis_t begin, axis_t end)
    : slice(type, input_shape, begin, end, axis_t(begin.size(), 1), 0, 0, 0, 0, 0)
{
}

bool slice::properties_equal(node &other) const
{
    auto &r = static_cast<slice &>(other);
    return begin() == r.begin() && end() == r.end() && strides() == r.strides()
        && begin_mask() == r.begin_mask() && end_mask() == r.end_mask() && ellipsis_mask() == r.ellipsis_mask()
        && new_axis_mask() == r.new_axis_mask() && shrink_axis_mask() == r.shrink_axis_mask();
}
