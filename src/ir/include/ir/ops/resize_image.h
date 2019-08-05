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
    class resize_image : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_resize_image);

        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        const std::array<int32_t, 2> &new_size() const noexcept { return new_size_; }
        image_resize_mode_t mode() const noexcept { return mode_; }
        bool align_corners() const noexcept { return align_corners_; }

        resize_image(datatype_t type, image_resize_mode_t mode, shape_t input_shape, std::array<int32_t, 2> new_size, bool align_corners);

    private:
        std::array<int32_t, 2> new_size_;
        image_resize_mode_t mode_;
        bool align_corners_;
    };
}
}
