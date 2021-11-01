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
#include <nncase/ir/ops/lstm.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_LSTM(const NodeProto &node)
{
    const auto &op_name { generate_name(node) };

    const auto &input = node.input()[0];
    const auto &W = node.input()[1];
    const auto &R = node.input()[2];
    const auto &B = node.input()[3];
    [[maybe_unused]] const auto &output = node.output()[0];
    [[maybe_unused]] const auto &output_h = node.output()[1];
    [[maybe_unused]] const auto &output_c = node.output()[2];
    [[maybe_unused]] const auto &sequence_lens = node.input()[4];
    auto static_shape = shape_t { 1, 1, 1, 1 };

    // 目前版本默认这俩为0
    // const auto &h_0 = node.input()[5];
    // const auto &c_0 = node.input()[6];

    // const datatype_t input_type = get_datatype(input).value();
    const auto &input_shape = get_shape(input);
    const auto &W_shape = get_shape(W);
    const auto &R_shape = get_shape(R);
    const auto &B_shape = get_shape(B);
    const auto &output_shape = get_shape(output);
    [[maybe_unused]] auto bias = get_constant_value<float>(B);

    std::vector<float> W_bias_vec { bias.begin(), bias.begin() + W_shape[1] };
    std::vector<float> R_bias_vec { bias.begin() + W_shape[1], bias.end() };

    auto W_bias = graph_.emplace<constant>(dt_float32, shape_t { W_shape[1] }, W_bias_vec);
    auto R_bias = graph_.emplace<constant>(dt_float32, shape_t { W_shape[1] }, R_bias_vec);

    // auto W_bitc = graph_.emplace<bitcast>(dt_float32, W_shape, shape_t { W_shape[1], W_shape[2] });
    // auto R_bitc = graph_.emplace<bitcast>(dt_float32, R_shape, shape_t { R_shape[1], R_shape[2] });
    // auto B_bitc = graph_.emplace<bitcast>(dt_float32, B_shape, shape_t { B_shape[1] });

    auto lstm_node = graph_.emplace<lstm>(input_shape, W_shape, W_bias->output().shape(), R_shape, R_bias->output().shape(), R_shape[2], false, "onnx");
    auto bitc_out = graph_.emplace<bitcast>(dt_float32, lstm_node->output().shape(), output_shape);

    bitc_out->input().connect(lstm_node->output());
    // lstm_node->w_xc().connect(W_bitc->output());
    lstm_node->b_xc().connect(W_bias->output());
    // lstm_node->w_rc().connect(R_bitc->output());
    lstm_node->b_rc().connect(R_bias->output());

    input_tensors_.emplace(&lstm_node->input_at(0), input);
    input_tensors_.emplace(&lstm_node->input_at(1), W);
    input_tensors_.emplace(&lstm_node->input_at(3), R);
    // lstm_node

    output_tensors_.emplace(output, &bitc_out->output());
}
