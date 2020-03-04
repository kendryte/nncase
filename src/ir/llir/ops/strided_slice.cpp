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
#include <hlir/op_utils.h>
#include <llir/ops/strided_slice.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::llir;

strided_slice::strided_slice(datatype_t type, shape_t input_shape, axis_t begin, axis_t end, axis_t strides, int32_t begin_mask, int32_t end_mask, int32_t ellipsis_mask, int32_t new_axis_mask, int32_t shrink_axis_mask)
    : begin_(hlir::normalize_strided_slice_begin(input_shape, begin, strides, begin_mask))
    , end_(hlir::normalize_strided_slice_end(input_shape, begin_, end, strides, end_mask, shrink_axis_mask))
    , strides_(std::move(strides))
    , begin_mask_(0)
    , end_mask_(0)
    , ellipsis_mask_(ellipsis_mask)
    , new_axis_mask_(new_axis_mask)
    , shrink_axis_mask_(shrink_axis_mask)
{
    add_input("input", type, input_shape);
    add_output("output", type, hlir::get_strided_slice_output_shape(begin_, end_, strides_, ellipsis_mask_, new_axis_mask_, shrink_axis_mask));
}
