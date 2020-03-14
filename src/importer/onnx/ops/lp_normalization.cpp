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
#include <hlir/ops/binary.h>
#include <hlir/ops/constant.h>
#include <hlir/ops/reduce.h>
#include <hlir/ops/unary.h>

// using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

void onnx_importer::convert_op_LpNormalization(const NodeProto& node)
{
    const auto &input { node.input()[0] };
    const auto &output { node.output()[0] };

    const auto &input_shape { get_shape(input) };

    axis_t reduce_axis { static_cast<int>(real_axis(get_attribute<int>(node, "axis").value(), input_shape.size())) };
    const auto p { get_attribute<int>(node, "p").value() };

    assert(p >= 1 && p <= 2);

    switch (p)
    {
    case 1:
    {
        auto abs = graph_.emplace<unary>(unary_abs, input_shape);
        auto sum = graph_.emplace<reduce>(reduce_sum, abs->output().shape(), reduce_axis, 0.f, true);

        sum->input().connect(abs->output());

        input_tensors_.emplace(&abs->input(), input);
        output_tensors_.emplace(output, &sum->output());
    }
    case 2:
    {
        auto square = graph_.emplace<unary>(unary_square, input_shape);
        auto sum = graph_.emplace<reduce>(reduce_sum, square->output().shape(), reduce_axis, 0.f, true);
        auto epsilon = graph_.emplace<constant>(1e-10f);
        auto max = graph_.emplace<binary>(binary_max, sum->output().shape(), epsilon->output().shape(), value_range<float>::full());
        auto rsqrt = graph_.emplace<unary>(unary_rsqrt, max->output().shape());
        auto mul = graph_.emplace<binary>(binary_mul, input_shape, rsqrt->output().shape(), value_range<float>::full());

        sum->input().connect(square->output());
        max->input_a().connect(sum->output());
        max->input_b().connect(epsilon->output());
        rsqrt->input().connect(max->output());
        mul->input_b().connect(rsqrt->output());

        input_tensors_.emplace(&square->input(), input);
        input_tensors_.emplace(&mul->input_a(), input);
        output_tensors_.emplace(output, &mul->output());
    }
    }
}
