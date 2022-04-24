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
#include "../caffe_importer.h"
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_CAFFE_LOWER(Softmax)
{
    // check if there are bn/scale/relu above
    std::string input_name = get_real_input_names(op)[0];

    auto &input = *output_tensors_.at(input_name);
    auto &param = op.softmax_param();

    axis_t reduce_axis;
    reduce_axis.push_back(param.axis());

    auto max = graph_.emplace<reduce>(reduce_max, dt_float32, input.shape(), reduce_axis, std::numeric_limits<float>::lowest(), true);
    max->name(op.name() + "/max");
    auto sub = graph_.emplace<binary>(binary_sub, dt_float32, input.shape(), max->output().shape(), value_range<float>::full());
    sub->name(op.name() + "/sub");
    auto exp = graph_.emplace<unary>(unary_exp, sub->output().shape());
    exp->name(op.name() + "/exp");
    auto sum = graph_.emplace<reduce>(reduce_sum, dt_float32, exp->output().shape(), reduce_axis, 0.f, true);
    sum->name(op.name() + "/sum");
    auto div = graph_.emplace<binary>(binary_div, dt_float32, exp->output().shape(), sum->output().shape(), value_range<float>::full());
    div->name(op.name() + "/div");

    sub->input_b().connect(max->output());
    exp->input().connect(sub->output());
    sum->input().connect(exp->output());
    div->input_a().connect(exp->output());
    div->input_b().connect(sum->output());

    input_tensors_.emplace(&max->input(), input_name);
    input_tensors_.emplace(&sub->input_a(), input_name);
    output_tensors_.emplace(op.top(0), &div->output());
}
