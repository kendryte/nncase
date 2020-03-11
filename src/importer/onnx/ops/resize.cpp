/* Copyright 2020 Alexey Chernov <4ernov@gmail.com>
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

#include "../onnx_importer.h"

#include <cassert>

#include <hlir/graph.h>
#include <hlir/ops/resize_image.h>

using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

namespace
{
    bool parse_align_corners(const string &value) noexcept
    {
        return value == "align_corners";
    }

    image_resize_mode_t parse_image_resize_mode(const string& value) noexcept
    {
        if (value == "linear")
            return image_resize_bilinear;
        else
            return image_resize_nearest_neighbor;
    }
}

void onnx_importer::convert_op_Resize(const NodeProto& node)
{
    const auto &input { node.input()[0] };
    const auto &size { node.input()[1] };
    const auto &output { node.output()[0] };

    const auto input_type { get_datatype(input).value() };
    const auto &input_shape { get_shape(input) };

    axis_t new_size_value { to<axis_t>(get_initializer(size)) };
    array<int32_t, 2> new_size { new_size_value[0], new_size_value[1] };

    const auto mode { parse_image_resize_mode(get_attribute<string>(node, "mode").value()) };
    const auto align_corners { parse_align_corners(get_attribute<string>(node, "coordinate_transformation_mode").value()) };

    auto op { graph_.emplace<resize_image>(input_type, mode, input_shape, new_size, align_corners) };

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}
