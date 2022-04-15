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
#pragma once
#include "../node.h"
#include <xtensor/xarray.hpp>

namespace nncase::ir
{
class NNCASE_API batch_to_space : public node
{
public:
    DEFINE_NODE_OPCODE(op_batch_to_space);

    input_connector &input() { return input_at(0); }
    output_connector &output() { return output_at(0); }

    int32_t block_size_h() const noexcept { return block_size_h_; }
    int32_t block_size_w() const noexcept { return block_size_w_; }
    int32_t real_block_size_h() const noexcept { return real_block_size_h_; }
    int32_t real_block_size_w() const noexcept { return real_block_size_w_; }

    const axis_t &begin() const noexcept { return begin_; }
    const axis_t &end() const noexcept { return end_; }
    const axis_t &strides() const noexcept { return strides_; }
    int32_t begin_mask() const noexcept { return begin_mask_; }
    int32_t end_mask() const noexcept { return end_mask_; }
    int32_t ellipsis_mask() const noexcept { return ellipsis_mask_; }
    int32_t new_axis_mask() const noexcept { return new_axis_mask_; }
    int32_t shrink_axis_mask() const noexcept { return shrink_axis_mask_; }
    std::array<int32_t, 2> crop_h() const noexcept { return crop_h_; }
    std::array<int32_t, 2> crop_w() const noexcept { return crop_w_; }

    batch_to_space(datatype_t input_type, shape_t input_shape, int32_t block_shape_h, int32_t block_shape_w, axis_t stride, axis_t begin, axis_t end, std::array<int32_t, 2> crop_h_, std::array<int32_t, 2> crop_w_, int32_t real_block_shape_h, int32_t real_block_shape_w);

protected:
    bool properties_equal(node &other) const override;

private:
    int32_t block_size_h_;
    int32_t block_size_w_;
    axis_t begin_;
    axis_t end_;
    axis_t strides_;
    int32_t begin_mask_;
    int32_t end_mask_;
    int32_t ellipsis_mask_;
    int32_t new_axis_mask_;
    int32_t shrink_axis_mask_;
    std::array<int32_t, 2> crop_h_;
    std::array<int32_t, 2> crop_w_;
    int32_t real_block_size_h_;
    int32_t real_block_size_w_;
};
}
