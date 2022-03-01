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
#include <nncase/ir/ops/sigmoid.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(LOGISTIC)
{
    auto &input = get_tensor(op.inputs(), 0);

    auto in_shape = get_shape(input.shape());
    auto input_type = to_data_type(input.type());

#if 0
    auto neg = graph_.emplace<unary>(unary_neg, in_shape);
    auto exp = graph_.emplace<unary>(unary_exp, neg->output().shape());
    auto one = graph_.emplace<constant>(1.f);
    auto plus = graph_.emplace<binary>(binary_add, input_type, one->output().shape(), exp->output().shape(), value_range<float>::full());
    auto div = graph_.emplace<binary>(binary_div, input_type, one->output().shape(), plus->output().shape(), value_range<float>::full());

    neg->name(get_tensor(op.outputs(), 0).name()->string_view());
    exp->name(get_tensor(op.outputs(), 0).name()->string_view());
    one->name(get_tensor(op.outputs(), 0).name()->string_view());
    plus->name(get_tensor(op.outputs(), 0).name()->string_view());
    div->name(get_tensor(op.outputs(), 0).name()->string_view());

    exp->input().connect(neg->output());
    plus->input_a().connect(one->output());
    plus->input_b().connect(exp->output());
    div->input_a().connect(one->output());
    div->input_b().connect(plus->output());

    link_input_tensor(&neg->input(), op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), &div->output());
#else
    auto sigmd = graph_.emplace<sigmoid>(input_type, in_shape);
    sigmd->name(get_tensor(op.outputs(), 0).name()->string_view());

    link_input_tensor(&sigmd->input(), op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), &sigmd->output());
#endif
}
