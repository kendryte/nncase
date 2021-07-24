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
#include <nncase/ir/ops/matmul.h>
#include <nncase/ir/ops/transpose.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Gemm(const NodeProto &node)
{
    const auto &op_name { generate_name(node) };

    assert(node.input().size() >= 2 && node.input().size() <= 3);
    assert(node.output().size() > 0 && node.output().size() <= 5);

    const auto alpha_attr = get_attribute<float>(node, "alpha");
    const float alpha_value = alpha_attr ? alpha_attr.value() : 1.0f;

    const auto transA_attr = get_attribute<int>(node, "transA");
    const bool transA = transA_attr ? static_cast<bool>(transA_attr.value()) : false;

    const auto transB_attr = get_attribute<int>(node, "transB");
    const bool transB = transB_attr ? static_cast<bool>(transB_attr.value()) : false;

    const auto &input_A = node.input()[0];
    const auto &input_B = node.input()[1];
    const auto &output = node.output()[0];

    const auto &add_transpose {
        [this](const auto &input)
        {
            const auto input_type = get_datatype(input).value();
            const auto &input_shape = get_shape(input);

            axis_t perm(input_shape.size());
            std::iota(std::begin(perm), std::end(perm), 0);
            std::reverse(std::begin(perm), std::end(perm));

            return graph_.emplace<transpose>(input_type, input_shape, perm);
        }
    };

    auto transA_op = transA ? add_transpose(input_A) : nullptr;
    if (transA_op)
        transA_op->name(op_name + ".transpose_A(Gemm)");

    auto transB_op = transB ? add_transpose(input_B) : nullptr;
    if (transB_op)
        transB_op->name(op_name + ".transpose_B(Gemm)");

    const auto &As_shape = transA ? transA_op->output().shape() : get_shape(input_A);
    const auto &Bs_shape = transB ? transB_op->output().shape() : get_shape(input_B);

    auto alpha = graph_.emplace<constant>(alpha_value);
    alpha->name(op_name + ".alpha(Gemm)");
    auto alpha_A_op = graph_.emplace<binary>(binary_mul, alpha->output().shape(), As_shape, value_range<float>::full());
    alpha_A_op->name(op_name + ".mul_A(Gemm)");
    auto A_B_op = graph_.emplace<matmul>(alpha_A_op->output().shape(), Bs_shape, value_range<float>::full());
    A_B_op->name(op_name + ".matmul(Gemm)");

    alpha_A_op->input_a().connect(alpha->output());
    A_B_op->input_a().connect(alpha_A_op->output());

    if (transA)
    {
        alpha_A_op->input_b().connect(transA_op->output());
        input_tensors_.emplace(&transA_op->input(), input_A);
    }
    else
    {
        input_tensors_.emplace(&alpha_A_op->input_b(), input_A);
    }

    if (transB)
    {
        A_B_op->input_b().connect(transB_op->output());
        input_tensors_.emplace(&transB_op->input(), input_B);
    }
    else
    {
        input_tensors_.emplace(&A_B_op->input_b(), input_B);
    }

    if (node.input().size() > 2)
    {
        const auto beta_attr = get_attribute<float>(node, "beta");
        const float beta_value = beta_attr ? beta_attr.value() : 1.0f;
        auto beta = graph_.emplace<constant>(beta_value);
        beta->name(op_name + ".beta(Gemm)");
        const auto &input_C = node.input()[2];
        auto beta_C_op = graph_.emplace<binary>(binary_mul, beta->output().shape(), get_shape(input_C), value_range<float>::full());
        beta_C_op->name(op_name + ".mul_C(Gemm)");

        beta_C_op->input_a().connect(beta->output());
        A_B_op->bias().connect(beta_C_op->output());

        input_tensors_.emplace(&beta_C_op->input_b(), input_C);
    }
    else
    {
        std::vector<float> bias_value(As_shape.back(), 0.f);
        shape_t bias_shape = { As_shape.back() };
        auto bias = graph_.emplace<constant>(dt_float32, bias_shape, bias_value);
        bias->name(op_name + ".bias(Gemm)");
        A_B_op->bias().connect(bias->output());
    }
    output_tensors_.emplace(output, &A_B_op->output());
}
