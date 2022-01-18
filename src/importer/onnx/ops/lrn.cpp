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
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/ops/unary.h>
#include <vector>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_LRN(const NodeProto &node)
{
    const auto &op_name { generate_name(node) };

    const std::string &input = node.input()[0];
    const auto input_type = get_datatype(input).value();
    const shape_t &input_shape = get_shape(input);

    // size
    auto size_value = get_attribute<int>(node, "size").value();

    // alpha
    auto alpha_value = get_attribute<float>(node, "alpha").value_or(0.0001);
    auto alpha = graph_.emplace<constant>(alpha_value / size_value);
    alpha->name(op_name + ".alpha(LRN)");

    // beta
    auto beta_value = get_attribute<float>(node, "beta").value_or(0.75);
    auto beta = graph_.emplace<constant>(beta_value);
    beta->name(op_name + ".beta(LRN)");

    // bias
    auto bias_value = get_attribute<float>(node, "bias").value_or(1.0);
    auto bias = graph_.emplace<constant>(bias_value);
    bias->name(op_name + ".bias(LRN)");

    auto square = graph_.emplace<unary>(unary_square, input_shape);
    square->name(op_name + ".square(LRN)");

    std::vector<ir::shape_t> concat_shape(input_shape[1], ir::shape_t { input_shape[0], 1, input_shape[2], input_shape[3] });
    auto con = graph_.emplace<concat>(input_type, concat_shape, 1);
    con->name(op_name + ".concat(LRN)");

    for (size_t i = 0; i < input_shape[1]; i++)
    {
        auto begin = std::max(static_cast<int32_t>(0), static_cast<int32_t>(i - std::floor((size_value - 1) / 2)));
        auto end = std::min(static_cast<int32_t>(input_shape[1] - 1), static_cast<int32_t>(i + std::ceil((size_value - 1) / 2)));

        auto sl = graph_.emplace<slice>(input_type, input_shape,
            axis_t { 0, begin, 0, 0 },
            axis_t { static_cast<int32_t>(input_shape[0]), static_cast<int32_t>(end + 1), static_cast<int32_t>(input_shape[2]), static_cast<int32_t>(input_shape[3]) },
            axis_t { 1, 1, 1, 1 }, 0, 0, 0, 0);
        sl->name(op_name + ".slice_" + std::to_string(i) + "(LRN)");
        sl->input().connect(square->output());

        auto r_sum = graph_.emplace<reduce>(reduce_sum, sl->output().shape(), axis_t { 1 }, 0.f, true);
        r_sum->name(op_name + ".reduce_sum_" + std::to_string(i) + "(LRN)");
        r_sum->input().connect(sl->output());
        con->input_at(i).connect(r_sum->output());
    }

    auto mul = graph_.emplace<binary>(binary_mul, input_type, alpha->output().shape(), con->output().shape(), value_range<float>::full());
    mul->name(op_name + ".mul(LRN)");
    auto add = graph_.emplace<binary>(binary_add, input_type, mul->output().shape(), bias->output().shape(), value_range<float>::full());
    add->name(op_name + ".add(LRN)");
    auto pow = graph_.emplace<binary>(binary_pow, input_type, add->output().shape(), beta->output().shape(), value_range<float>::full());
    pow->name(op_name + ".pow(LRN)");
    auto div = graph_.emplace<binary>(binary_div, input_type, input_shape, pow->output().shape(), value_range<float>::full());
    div->name(op_name + ".div(LRN)");

    mul->input_a().connect(alpha->output());
    mul->input_b().connect(con->output());
    add->input_a().connect(mul->output());
    add->input_b().connect(bias->output());
    pow->input_a().connect(add->output());
    pow->input_b().connect(beta->output());
    div->input_b().connect(pow->output());

    input_tensors_.emplace(&square->input(), input);
    input_tensors_.emplace(&div->input_a(), input);
    output_tensors_.emplace(node.output()[0], &div->output());
}
