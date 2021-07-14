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
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_BatchNormalization(const NodeProto &node)
{
    assert(node.input().size() == 5);
    assert(node.output().size() > 0 && node.output().size() <= 5);

    auto epsilon_attr = get_attribute<float>(node, "epsilon");
    auto epsilon = !epsilon_attr ? 1e-05f : epsilon_attr.value();

    const auto &op_name { generate_name(node) };

    const auto &input_T = node.input()[0];
    const auto &input_scale = node.input()[1];
    const auto &input_B = node.input()[2];
    const auto &input_mean = node.input()[3];
    const auto &input_var = node.input()[4];

    const auto &output_T = node.output()[0];

    const auto &input_T_shape = get_shape(input_T);

    const auto &broadcast_if_needed {
        [this, &input_T_shape](const std::string &input) -> ir::bitcast * {
            const auto &input_shape { get_shape(input) };

            if (input_T_shape.empty() || input_shape.size() == input_T_shape.size() - 1)
                return nullptr;

            const auto &target_shape { broadcast_shape(input_shape, input_T_shape) };
            std::cout << "input " << input << " target shape: [";
            for (auto s : target_shape)
                std::cout << ' ' << s;
            std::cout << ']' << std::endl;
            auto reshape_op = graph_.emplace<bitcast>(get_datatype(input).value(), input_shape, target_shape);

            input_tensors_.emplace(&reshape_op->input(), input);

            return reshape_op;
        }
    };

    auto pre_mean_op = broadcast_if_needed(input_mean);
    pre_mean_op->name(op_name + ".broadcast_mean(BatchNormalization)");
    auto mean_op = graph_.emplace<binary>(binary_sub, input_T_shape, pre_mean_op ? pre_mean_op->output().shape() : get_shape(input_mean), value_range<float>::full());
    mean_op->name(op_name + ".mean(BatchNormalization)");

    auto eps = graph_.emplace<constant>(epsilon);
    eps->name(op_name + ".eps(BatchNormalization)");

    auto pre_var_op = broadcast_if_needed(input_var);
    pre_var_op->name(op_name + ".broadcast_var(BatchNormalization)");
    auto eps_op = graph_.emplace<binary>(binary_add, pre_var_op ? pre_var_op->output().shape() : get_shape(input_var), eps->output().shape(), value_range<float>::full());
    eps_op->name(op_name + ".var_stab(BatchNormalization)");

    auto var_denom_op = graph_.emplace<unary>(unary_rsqrt, eps_op->output().shape());
    var_denom_op->name(op_name + ".var_denom(BatchNormalization)");

    auto norm_op = graph_.emplace<binary>(binary_mul, mean_op->output().shape(), var_denom_op->output().shape(), value_range<float>::full());
    norm_op->name(op_name + ".norm(BatchNormalization)");

    auto pre_scale_op = broadcast_if_needed(input_scale);
    pre_scale_op->name(op_name + ".broadcast_scale(BatchNormalization)");
    auto scale_op = graph_.emplace<binary>(binary_mul, norm_op->output().shape(), pre_scale_op ? pre_scale_op->output().shape() : get_shape(input_scale), value_range<float>::full());
    scale_op->name(op_name + ".scale(BatchNormalization)");

    auto pre_B_op = broadcast_if_needed(input_B);
    pre_B_op->name(op_name + ".broadcast_bias(BatchNormalization)");
    auto bias_op = graph_.emplace<binary>(binary_add, scale_op->output().shape(), pre_B_op ? pre_B_op->output().shape() : get_shape(input_B), value_range<float>::full());
    bias_op->name(op_name + ".bias(BatchNormalization)");

    if (pre_mean_op)
        mean_op->input_b().connect(pre_mean_op->output());
    else
        input_tensors_.emplace(&mean_op->input_b(), input_mean);

    if (pre_var_op)
        eps_op->input_a().connect(pre_var_op->output());
    else
        input_tensors_.emplace(&eps_op->input_a(), input_var);

    eps_op->input_b().connect(eps->output());
    var_denom_op->input().connect(eps_op->output());
    norm_op->input_a().connect(mean_op->output());
    norm_op->input_b().connect(var_denom_op->output());
    scale_op->input_a().connect(norm_op->output());

    if (pre_scale_op)
        scale_op->input_b().connect(pre_scale_op->output());
    else
        input_tensors_.emplace(&scale_op->input_b(), input_scale);

    bias_op->input_a().connect(scale_op->output());

    if (pre_B_op)
        bias_op->input_b().connect(pre_B_op->output());
    else
        input_tensors_.emplace(&bias_op->input_b(), input_B);

    if (node.output().size() > 1)
        passthrough_connections_.emplace(node.output()[1], input_mean);
    if (node.output().size() > 2)
        passthrough_connections_.emplace(node.output()[2], input_var);
    if (node.output().size() > 3)
        passthrough_connections_.emplace(node.output()[3], input_mean);
    if (node.output().size() > 4)
        passthrough_connections_.emplace(node.output()[4], input_var);

    input_tensors_.emplace(&mean_op->input_a(), input_T);
    output_tensors_.emplace(output_T, &bias_op->output());
}
