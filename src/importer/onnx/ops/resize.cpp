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
    const bool use_version_9 { node.input().size() == 2 };

    const auto &input { node.input()[0] };
    const auto &scale { use_version_9 ? node.input()[1] : node.input()[2] };
    const auto &output { node.output()[0] };

    const auto input_type { get_datatype(input).value() };
    const auto &input_shape { get_shape(input) };

    axis_t new_size;

    if (!use_version_9 && node.input().size() == 4)
    {
        const auto &size { node.input()[3] };
        const auto &new_size_initializer { get_initializer(size) };

        if (new_size_initializer)
        {
            new_size = to<axis_t>(new_size_initializer.value());
        }
        else
        {
            // try to extract data from previous constant nodes
            const auto data { get_constant_input_data<float>(size) };

            if (data)
                transform(begin(data.value()), end(data.value()), back_inserter(new_size),
                    [](const auto e) { return static_cast<int>(e); });
        }
    }

    if (new_size.empty())
    {
        // try to extract data from previous constant nodes
        const auto data { get_constant_input_data<float>(scale) };

        if (data)
            transform(begin(data.value()), end(data.value()), begin(input_shape), back_inserter(new_size),
                [](const auto axis_size, const auto scale) { return static_cast<int>(round(axis_size * scale)); });
    }

    const auto mode { parse_image_resize_mode(get_attribute<string>(node, "mode").value()) };
    const auto align_corners { parse_align_corners(get_attribute<string>(node, "coordinate_transformation_mode").value()) };
    const array<int32_t, 2> new_size_array {{ new_size[2], new_size[3] }};

    auto op { graph_.emplace<resize_image>(input_type, mode, input_shape, new_size_array, align_corners) };

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}
