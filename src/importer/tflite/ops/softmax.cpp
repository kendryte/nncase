/* Copyright 2019 Canaan Inc.
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
#include <ir/ops/binary.h>
#include <ir/ops/constant.h>
#include <ir/ops/reduce.h>
#include <ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(SOFTMAX)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &options = *op.builtin_options_as_SoftmaxOptions();

    auto in_shape = get_shape(input.shape());
    axis_t reduce_axis;
    reduce_axis.push_back(in_shape.size() - 1);

    auto max = graph_.emplace<reduce>(reduce_max, in_shape, reduce_axis, std::numeric_limits<float>::lowest(), true);
    auto sub = graph_.emplace<binary>(binary_sub, in_shape, max->output().shape(), value_range<float>::full());
    auto beta = graph_.emplace<constant>(options.beta());
    auto mul = graph_.emplace<binary>(binary_mul, sub->output().shape(), beta->output().shape(), value_range<float>::full());
    auto exp = graph_.emplace<unary>(unary_exp, mul->output().shape());
    auto sum = graph_.emplace<reduce>(reduce_sum, exp->output().shape(), reduce_axis, 0.f, true);
    auto div = graph_.emplace<binary>(binary_div, exp->output().shape(), sum->output().shape(), value_range<float>::full());

    sub->input_b().connect(max->output());
    mul->input_a().connect(sub->output());
    mul->input_b().connect(beta->output());
    exp->input().connect(mul->output());
    sum->input().connect(exp->output());
    div->input_a().connect(exp->output());
    div->input_b().connect(sum->output());

    input_tensors_.emplace(&max->input(), op.inputs()->Get(0));
    input_tensors_.emplace(&sub->input_a(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &div->output());
}
