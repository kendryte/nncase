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
#include <hlir/ops/clamp.h>
#include <hlir/ops/constant.h>
#include <hlir/ops/reduce.h>
#include <hlir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;
using namespace caffe;

DEFINE_CAFFE_LOWER(ReLU)
{
    auto &input = *output_tensors_.at(op.bottom(0));
    auto &param = op.relu_param();

    auto zero = graph_.emplace<constant>(0.f);
    auto high = graph_.emplace<constant>(std::numeric_limits<float>::max());
    auto cl = graph_.emplace<clamp>(input.shape(), zero->output().shape(), high->output().shape());

    cl->input_low().connect(zero->output());
    cl->input_high().connect(high->output());
    input_tensors_.emplace(&cl->input(), op.bottom(0));
    output_tensors_.emplace(op.top(0), &cl->output());
}
