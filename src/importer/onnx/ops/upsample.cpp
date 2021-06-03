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
#include <nncase/ir/graph.h>
#include <nncase/ir/ops/resize_image.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

namespace
{
image_resize_mode_t parse_image_resize_mode(const std::string &value) noexcept
{
    if (value == "linear")
        return image_resize_bilinear;
    else
        return image_resize_nearest_neighbor;
}
}

void onnx_importer::convert_op_Upsample(const NodeProto &node)
{
    const auto &input = node.input()[0];
    const auto &output = node.output()[0];

    const auto input_type = get_datatype(input).value();
    const auto &input_shape = get_shape(input);

    axis_t new_shape;
    if (node.input().size() == 2)
    {
        // version 9
        std::vector<float> scales;
        auto initializer = get_initializer(node.input()[1]);
        scales = initializer ? to<std::vector<float>>(initializer.value()) : get_constant_input_data<float>(node.input()[1]).value();

        std::transform(input_shape.begin(), input_shape.end(), scales.begin(), std::back_inserter(new_shape),
            [](const auto axis, const auto scale) { return static_cast<int>(std::floor(axis * scale)); });
    }
    else if (auto scales = get_attribute<std::vector<float>>(node, "scales"))
    {
        // version 7
        std::transform(input_shape.begin(), input_shape.end(), scales.value().begin(), std::back_inserter(new_shape),
            [](const auto axis, const auto scale) { return static_cast<int>(std::floor(axis * scale)); });
    }
    else if (auto height_scale = get_attribute<std::vector<float>>(node, "height_scale"))
    {
        // version 1
        auto width_scale = get_attribute<std::vector<float>>(node, "width_scale");
        std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(new_shape),
            [](size_t axis) { return static_cast<int32_t>(axis); });
        auto it = new_shape.rbegin();
        *it = static_cast<int>(std::floor(*it * width_scale.value()[0]));
        it++;
        *it = static_cast<int>(std::floor(*it * height_scale.value()[0]));
    }
    else
    {
        std::cerr << "invalid upsampling op version" << std::endl;
        std::abort();
    }

    const auto mode = parse_image_resize_mode(get_attribute<std::string>(node, "mode").value());
    const auto align_corners = true;
    std::array<int32_t, 2> new_size;
    auto it = new_shape.rbegin();
    new_size[1] = *it++;
    new_size[0] = *it;

    auto op = graph_.emplace<resize_image>(input_type, mode, input_shape, new_size, align_corners);

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}