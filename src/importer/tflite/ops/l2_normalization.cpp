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
#include "../tflite_importer.h"
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(L2_NORMALIZATION)
{
    auto &input = get_tensor(op.inputs(), 0);
    [[maybe_unused]] auto &options = *op.builtin_options_as_L2NormOptions();

    auto in_shape = get_shape(input.shape());
    auto input_type = to_data_type(input.type());
    axis_t reduce_axis;
    if (in_shape.size() == 1)
    {
        reduce_axis.push_back(0);
    }
    else
    {
        for (size_t i = 1; i < in_shape.size(); i++)
            reduce_axis.push_back(int32_t(i));
    }

    auto square = graph_.emplace<unary>(unary_square, in_shape);
    auto sum = graph_.emplace<reduce>(reduce_sum, input_type, square->output().shape(), reduce_axis, 0.f, true);
    auto epsilon = graph_.emplace<constant>(1e-10f);
    auto max = graph_.emplace<binary>(binary_max, input_type, sum->output().shape(), epsilon->output().shape(), value_range<float>::full());
    auto rsqrt = graph_.emplace<unary>(unary_rsqrt, max->output().shape());
    auto mul = graph_.emplace<binary>(binary_mul, input_type, in_shape, rsqrt->output().shape(), value_range<float>::full());

    square->name(get_tensor(op.outputs(), 0).name()->string_view());
    sum->name(get_tensor(op.outputs(), 0).name()->string_view());
    epsilon->name(get_tensor(op.outputs(), 0).name()->string_view());
    max->name(get_tensor(op.outputs(), 0).name()->string_view());
    rsqrt->name(get_tensor(op.outputs(), 0).name()->string_view());
    mul->name(get_tensor(op.outputs(), 0).name()->string_view());

    sum->input().connect(square->output());
    max->input_a().connect(sum->output());
    max->input_b().connect(epsilon->output());
    rsqrt->input().connect(max->output());
    mul->input_b().connect(rsqrt->output());

    link_input_tensor(&square->input(), op.inputs()->Get(0));
    link_input_tensor(&mul->input_a(), op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), &mul->output());
}
