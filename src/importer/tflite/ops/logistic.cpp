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

DEFINE_TFLITE_LOWER(LOGISTIC)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &options = *op.builtin_options_as_SoftmaxOptions();

    auto in_shape = get_shape(input.shape());

    auto neg = graph_.emplace<unary>(unary_neg, in_shape);
    auto exp = graph_.emplace<unary>(unary_exp, neg->output().shape());
    auto one = graph_.emplace<constant>(1.f);
    auto plus = graph_.emplace<binary>(binary_add, one->output().shape(), exp->output().shape(), value_range<float>::full());
    auto div = graph_.emplace<binary>(binary_div, one->output().shape(), plus->output().shape(), value_range<float>::full());

    exp->input().connect(neg->output());
    plus->input_a().connect(one->output());
    plus->input_b().connect(exp->output());
    div->input_a().connect(one->output());
    div->input_b().connect(plus->output());

    input_tensors_.emplace(&neg->input(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &div->output());
}
