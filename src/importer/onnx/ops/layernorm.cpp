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

#include "nncase/ir/ops/layernorm.h"
#include "../onnx_importer.h"
#include "nncase/ir/ir_types.h"
#include <cassert>
#include <nncase/ir/graph.h>
#include <nncase/ir/ops/constant.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_LayerNormalization(const NodeProto &node)
{
    assert(node.input().size() >= 2);

    const auto &op_name { generate_name(node) };

    const auto &input = node.input()[0];
    const auto &scale = node.input()[1];
    const auto output = node.output()[0];

    auto input_shape = get_shape(input);
    const auto input_type = get_datatype(input).value();

    std::vector<float> scale_value;
    auto scale_initializer = get_initializer(scale);
    scale_value = scale_initializer ? to<std::vector<float>>(scale_initializer.value()) : get_constant_input_data<float>(scale).value();
    auto scale_shape = get_shape(scale);
    auto scale_constant = graph_.emplace<constant>(get_datatype<float>(), scale_shape, scale_value);
    scale_constant->name(op_name + ".scale(LayerNormalization)");

    auto bias_shape = scale_shape;
    std::vector<float> bias_value(xt::compute_size(scale_shape), 0.f);
    if (node.input().size() > 2)
    {
        const auto &bias = node.input()[2];
        auto bias_initializer = get_initializer(bias);
        bias_value = bias_initializer ? to<std::vector<float>>(bias_initializer.value()) : get_constant_input_data<float>(bias).value();
    }
    auto bias_constant = graph_.emplace<constant>(get_datatype<float>(), bias_shape, bias_value);
    bias_constant->name(op_name + ".bias(LayerNormalization)");

    auto axis_attr = get_attribute<int>(node, "axis");
    int32_t axis = axis_attr ? axis_attr.value() : -1;

    auto epsilon_attr = get_attribute<float>(node, "epsilon");
    auto epsilon = epsilon_attr ? epsilon_attr.value() : 1e-05f;

    auto ln = graph_.emplace<layernorm>(input_type, input_shape, axis, epsilon);
    ln->name(op_name + ".layer_norm(LayerNormalization)");

    input_tensors_.emplace(&ln->input(), input);
    ln->scale().connect(scale_constant->output());
    ln->bias().connect(bias_constant->output());
    output_tensors_.emplace(output, &ln->output());
}
