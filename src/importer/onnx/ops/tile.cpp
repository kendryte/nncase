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
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/slice.h>
#include <xtensor/xadapt.hpp>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Tile(const NodeProto &node)
{
    assert(node.input().size() >= 2);
    assert(node.output().size() == 1);
    const auto &op_name { generate_name(node) };

    // input
    const auto &input = node.input()[0];
    const datatype_t input_type = get_datatype(input).value();
    auto input_shape = get_shape(input);
    auto input_shape_size = input_shape.size();

    // output
    const auto &output = node.output()[0];

    // repeats
    std::vector<int32_t> repeats(input_shape_size, 1);

    // opset 1
    if (node.input().size() == 3)
    {
        // tile
        auto tile = get_constant_value<float, float>(node.input()[1]);
        assert(tile.size() == 1);

        // axis
        auto axes = get_constant_value<float, float>(node.input()[2]);
        assert(axes.size() == 1);
        auto axis = real_axis(static_cast<int32_t>(axes[0]), input_shape_size);

        repeats[axis] = static_cast<int32_t>(tile[0]);
    }
    else
    {
        // opset 6/11
        repeats = get_constant_value<int32_t, int64_t>(node.input()[1]);
        assert(repeats.size() == input_shape_size);
    }

    auto cur_shape = input_shape;
    concat *pcat = nullptr;
    std::vector<shape_t> concat_shapes;
    int32_t end = repeats.size() - 1;
    for (auto i = end; i >= 0; i--)
    {
        // skip the repeats[i] == 1
        if (repeats[i] == 1)
            continue;

        concat_shapes.clear();
        concat_shapes.assign(repeats[i], cur_shape);
        auto cat = graph_.emplace<concat>(input_type, concat_shapes, i);
        cat->name(op_name + ".concat(axis_" + std::to_string(i) + ")");
        if (pcat == nullptr)
        {
            for (auto j = 0; j < repeats[i]; j++)
                input_tensors_.emplace(&cat->input_at(j), input);
        }
        else
        {
            for (auto j = 0; j < repeats[i]; j++)
                cat->input_at(j).connect(pcat->output());
        }

        pcat = cat;
        cur_shape = pcat->output().shape();
    }

    if (pcat == nullptr)
    {
        // For all_of(repeats) == 1, just reshape the input tensor
        auto bc = graph_.emplace<bitcast>(input_type, input_shape, input_shape);
        bc->name(op_name + "/reshape");
        input_tensors_.emplace(&bc->input(), input);
        output_tensors_.emplace(output, &bc->output());
    }
    else
    {
        output_tensors_.emplace(output, &pcat->output());
    }
}
