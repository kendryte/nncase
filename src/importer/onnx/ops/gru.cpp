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
#include <nncase/ir/ops/gru.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_GRU(const NodeProto &node)
{
    const auto &op_name { generate_name(node) };

    // attribute
    auto direction_str = get_attribute<std::string>(node, "direction").value_or("forward");
    lstm_direction direction = kForward;
    if (direction_str == "forward")
        direction = kForward;
    else if (direction_str == "reverse")
        direction = kReverse;
    else
        direction = kBidirectional;
    size_t num_directions = direction == kBidirectional ? 2 : 1;

    auto linear_before_reset = get_attribute<int64_t>(node, "linear_before_reset").value_or(0);

    // input
    auto input_size = node.input_size();
    assert(input_size >= 3 && input_size <= 8);
    const auto &input = node.input()[0];
    const auto &W = node.input()[1];
    const auto &R = node.input()[2];

    const datatype_t input_type = get_datatype(input).value();
    const auto &input_shape = get_shape(input);
    const auto &W_shape = get_shape(W);
    const auto &R_shape = get_shape(R);

    size_t seq_length = input_shape[0];
    size_t batch_size = input_shape[1];
    size_t hidden_size = get_attribute<std::int64_t>(node, "hidden_size").value_or(W_shape[1] / 3);

    // bias
    std::string B;
    shape_t B_shape { num_directions, 6 * hidden_size };
    if (input_size >= 4)
    {
        B = node.input()[3];
    }

    // initial_h
    std::string initial_h;
    shape_t initial_shape { num_directions, batch_size, hidden_size };
    if (input_size >= 6)
    {
        initial_h = node.input()[5];
    }

    // output
    auto output_size = node.output_size();
    assert(output_size >= 0 && output_size <= 3);
    std::string output;
    if (output_size >= 1)
        output = node.output()[0];

    std::string output_h;
    if (output_size >= 2)
        output_h = node.output()[1];

    shape_t output_shape { seq_length, num_directions, batch_size, hidden_size };
    auto lstm_node = graph_.emplace<gru>(input_shape, W_shape, R_shape, B_shape, output_shape, initial_shape, direction, "onnx", linear_before_reset == 0 ? false : true);
    lstm_node->name(op_name);

    input_tensors_.emplace(&lstm_node->input_at(0), input);
    input_tensors_.emplace(&lstm_node->input_at(1), W);
    input_tensors_.emplace(&lstm_node->input_at(2), R);
    if (!B.empty())
    {
        input_tensors_.emplace(&lstm_node->input_at(3), B);
    }
    else
    {
        std::vector<float> v(xt::compute_size(B_shape), 0.f);
        auto c = graph_.emplace<constant>(input_type, B_shape, v);
        lstm_node->b().connect(c->output());
    }

    if (!initial_h.empty())
    {
        input_tensors_.emplace(&lstm_node->input_at(4), initial_h);
    }
    else
    {
        std::vector<float> v(xt::compute_size(initial_shape), 0.f);
        auto c = graph_.emplace<constant>(input_type, initial_shape, v);
        lstm_node->initial_h().connect(c->output());
    }

    if (!output.empty())
        output_tensors_.emplace(output, &lstm_node->output());

    if (!output_h.empty())
        output_tensors_.emplace(output_h, &lstm_node->output_h());
}
