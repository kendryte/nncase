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
#include <hlir/ops/constant.h>
#include <hlir/ops/binary.h>
#include <hlir/ops/matmul.h>
#include <hlir/ops/transpose.h>

using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

void onnx_importer::convert_op_Gemm(const NodeProto &node)
{
    assert(node.input().size() >= 2 && node.input().size() <= 3);
    assert(node.output().size() > 0 && node.output().size() <= 5);

    const auto alpha_attr { get_attribute<float>(node, "alpha") };
    const float alpha { alpha_attr ? alpha_attr.value() : 1.0f };

    const auto beta_attr { get_attribute<float>(node, "beta") };
    const float beta { beta_attr ? beta_attr.value() : 1.0f };

	const auto transA_attr { get_attribute<int>(node, "transA") };
	const bool transA { transA_attr ? static_cast<bool>(transA_attr.value()) : false };

	const auto transB_attr { get_attribute<int>(node, "transB") };
	const bool transB { transB_attr ? static_cast<bool>(transB_attr.value()) : false };

    const auto &input_A { node.input()[0] };
    const auto &input_B { node.input()[1] };
    const auto input_B_type { get_datatype(input_B).value() };

    const auto &output { node.output()[0] };

	const auto& add_transpose
	{
		[this](const auto& input)
		{
			const auto input_type { get_datatype(input).value() };
			const auto &input_shape { get_shape(input) };

			axis_t perm(input_shape.size());
			std::iota(begin(perm), end(perm), 0);
			std::reverse(begin(perm), end(perm));

			return graph_.emplace<transpose>(input_type, input_shape, perm);
		}
	};

	auto transA_op { transA ? add_transpose(input_A) : nullptr };
	auto transB_op { transB ? add_transpose(input_B) : nullptr };

	const auto &As_shape { transA ? transA_op->output().shape() : get_shape(input_A) };
	const auto &Bs_shape { transB ? transB_op->output().shape() : get_shape(input_B) };

    const auto &bias(xt::zeros<float>({ As_shape.back() }));

	auto alpha_constant { graph_.emplace<constant>(alpha) };
    auto alpha_A_op { graph_.emplace<binary>(binary_mul, alpha_constant->output().shape(), As_shape, value_range<float>::full()) };
    auto A_B_op { graph_.emplace<matmul>(alpha_A_op->output().shape(), Bs_shape, bias, value_range<float>::full()) };

	if (node.input().size() > 2)
	{
		const auto &input_C { node.input()[2] };
		auto beta_constant { graph_.emplace<constant>(beta) };
		auto beta_C_op { graph_.emplace<binary>(binary_mul, beta_constant->output().shape(), get_shape(input_C), value_range<float>::full()) };
		auto A_B_C_op { graph_.emplace<binary>(binary_add, A_B_op->output().shape(), beta_C_op->output().shape(), value_range<float>::full()) };

		beta_C_op->input_a().connect(beta_constant->output());
		A_B_C_op->input_a().connect(A_B_op->output());
		A_B_C_op->input_b().connect(beta_C_op->output());

		input_tensors_.emplace(&beta_C_op->input_b(), input_C);

		output_tensors_.emplace(output, &A_B_C_op->output());
	}
	else
	{
		output_tensors_.emplace(output, &A_B_op->output());
	}

	alpha_A_op->input_a().connect(alpha_constant->output());
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

}
