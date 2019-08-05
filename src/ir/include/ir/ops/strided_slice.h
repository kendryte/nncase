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
#pragma once
#include "../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace ir
{
    class strided_slice : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_strided_slice);

        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        const axis_t &begin() const noexcept { return begin_; }
        const axis_t &end() const noexcept { return end_; }
        const axis_t &strides() const noexcept { return strides_; }
        int32_t begin_mask() const noexcept { return begin_mask_; }
        int32_t end_mask() const noexcept { return end_mask_; }
        int32_t ellipsis_mask() const noexcept { return ellipsis_mask_; }
        int32_t new_axis_mask() const noexcept { return new_axis_mask_; }
        int32_t shrink_axis_mask() const noexcept { return shrink_axis_mask_; }

        strided_slice(datatype_t type, shape_t input_shape, axis_t begin, axis_t end, axis_t strides, int32_t begin_mask, int32_t end_mask, int32_t ellipsis_mask, int32_t new_axis_mask, int32_t shrink_axis_mask);

    private:
        axis_t begin_;
        axis_t end_;
        axis_t strides_;
        int32_t begin_mask_;
        int32_t end_mask_;
        int32_t ellipsis_mask_;
        int32_t new_axis_mask_;
        int32_t shrink_axis_mask_;
    };
}
}
