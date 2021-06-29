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
#include <nncase/ir/ops/resize_image.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

resize_image::resize_image(datatype_t type, image_resize_mode_t mode, shape_t input_shape, std::array<int32_t, 2> new_size, bool align_corners)
    : new_size_(new_size), mode_(mode), align_corners_(align_corners)
{
    add_input("input", type, input_shape);
    add_output("output", type, get_resize_image_shape(input_shape, new_size));
}

bool resize_image::properties_equal(node &other) const
{
    auto &r = static_cast<resize_image &>(other);
    return mode() == r.mode() && new_size() == r.new_size() && align_corners() == r.align_corners();
}
