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
#include <nncase/ir/ops/clamp.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace caffe;

DEFINE_CAFFE_LOWER(ReLU)
{
    // check if there are bn/scale/relu above
    std::string input_name = get_real_input_names(op)[0];

    auto &input = *output_tensors_.at(input_name);

    auto &param = op.relu_param();
    if (param.negative_slope() != 0)
        throw std::runtime_error("negative slope is not supported");

    auto zero = graph_.emplace<constant>(0.f);
    zero->name(op.name() + "/zero_const");
    auto high = graph_.emplace<constant>(std::numeric_limits<float>::max());
    high->name(op.name() + "/high_const");
    auto cl = graph_.emplace<clamp>(input.shape(), zero->output().shape(), high->output().shape());
    if (op.bottom(0) == op.top(0))
    {
        // inplace op, user op need this name
        cl->name(op.top(0) + "/clamp");
    }
    else
        cl->name(op.name() + "/clamp");

    cl->input_low().connect(zero->output());
    cl->input_high().connect(high->output());

    input_tensors_.emplace(&cl->input(), input_name);
    if (op.bottom(0) == op.top(0))
    {
        // inplace op, user op need this name
        output_tensors_.emplace(cl->name(), &cl->output());
    }
    else
        output_tensors_.emplace(op.top(0), &cl->output());
}
