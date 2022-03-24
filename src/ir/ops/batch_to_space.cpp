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
#include <nncase/ir/ops/batch_to_space.h>

using namespace nncase;
using namespace nncase::ir;

batch_to_space::batch_to_space(datatype_t input_type, shape_t input_shape, int32_t block_shape_h, int32_t block_shape_w, axis_t strides, axis_t begin, axis_t end, std::array<int32_t, 2> crop_h, std::array<int32_t, 2> crop_w, int32_t real_block_shape_h, int32_t real_block_shape_w)
    : block_size_h_(block_shape_h), block_size_w_(block_shape_w), begin_(normalize_strided_slice_begin(input_shape, begin, strides, 0)), end_(normalize_strided_slice_end(input_shape, begin_, end, strides, 0, 0)), strides_(std::move(strides)), begin_mask_(0), end_mask_(0), ellipsis_mask_(0), new_axis_mask_(0), shrink_axis_mask_(0), crop_h_(crop_h), crop_w_(crop_w), real_block_size_h_(real_block_shape_h), real_block_size_w_(real_block_shape_w)
{
    add_input("input", input_type, input_shape);
    add_output("output", input_type,
        shape_t {
            input_shape[0] / (real_block_size_h_ * real_block_size_w_),
            input_shape[1],
            get_strided_slice_output_shape(begin, end_, strides_, 0, 0, 0)[1],
            get_strided_slice_output_shape(begin, end_, strides_, 0, 0, 0)[2],
        });
}

bool batch_to_space::properties_equal(node &other) const
{
    auto &r = static_cast<batch_to_space &>(other);
    return begin() == r.begin() && end() == r.end() && strides() == r.strides()
        && begin_mask() == r.begin_mask() && end_mask() == r.end_mask() && ellipsis_mask() == r.ellipsis_mask()
        && new_axis_mask() == r.new_axis_mask() && shrink_axis_mask() == r.shrink_axis_mask()
        && crop_h() == r.crop_h() && crop_w() == r.crop_w()
        && real_block_size_h() == r.real_block_size_h() && real_block_size_w() == r.real_block_size_w();
}
