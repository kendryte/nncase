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

void onnx_importer::convert_op_Resize(const NodeProto &node)
{
    const auto &op_name { generate_name(node) };

    const auto &input = node.input()[0];
    const bool use_version_10 = (node.input().size() == 2) && (get_datatype(node.input()[1]).value() == dt_float32);
    const auto &scale = use_version_10 ? node.input()[1] : node.input()[2];
    const auto &output = node.output()[0];

    const auto input_type = get_datatype(input).value();
    const auto &input_shape = get_shape(input);

    axis_t new_size;

    // use scale
    std::vector<float> scales;
    const auto &initializer = get_initializer(scale);
    if (initializer)
    {
        scales = to<std::vector<float>>(initializer.value());
    }
    else
    {
        // try to extract data from previous constant nodes
        const auto data = get_constant_input_data<float>(scale);
        if (data)
            scales = data.value();
    }

    if (!scales.empty())
    {
        std::transform(std::begin(input_shape), std::end(input_shape), std::begin(scales), std::back_inserter(new_size),
            [](const auto dim, const auto scale) { return static_cast<int>(std::floor(dim * scale)); });
    }

    // use size
    if (new_size.empty())
    {
        assert(node.input().size() == 4);
        const auto &size = node.input()[3];
        const auto &initializer = get_initializer(size);
        if (initializer)
        {
            new_size = to<axis_t>(initializer.value());
        }
        else
        {
            // try to extract data from previous constant nodes
            const auto data = get_constant_input_data<float>(size);
            if (data)
                std::transform(std::begin(data.value()), std::end(data.value()), std::back_inserter(new_size),
                    [](const auto e) { return static_cast<int>(e); });
        }
    }

    assert(!new_size.empty());

    auto mode_attr = get_attribute<std::string>(node, "mode");
    const auto mode = parse_image_resize_mode(!mode_attr ? "nearest" : mode_attr.value());

    auto ct_attr = get_attribute<std::string>(node, "coordinate_transformation_mode");
    auto ct_attr_value = !ct_attr ? "asymmetric" : ct_attr.value();
    const auto pytorch_half_pixel = ct_attr_value == "pytorch_half_pixel";
    const auto align_corners = ct_attr_value == "align_corners";

    const std::array<int32_t, 2> new_size_array = { new_size[2], new_size[3] };

    auto op = graph_.emplace<resize_image>(input_type, mode, input_shape, new_size_array, align_corners, pytorch_half_pixel);
    op->name(op_name + "(Resize)");

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}
