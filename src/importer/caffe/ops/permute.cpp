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
#include <nncase/ir/ops/transpose.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace caffe;

DEFINE_CAFFE_LOWER(Permute)
{
    // check if there are bn/scale/relu above
    std::string input_name = get_real_input_names(op)[0];

    auto &input = *output_tensors_.at(input_name);
    auto &param = op.permute_param();
    axis_t perm;
    for (int i = 0; i < param.order_size(); i++)
        perm.push_back(param.order(i));

    auto tp = graph_.emplace<transpose>(dt_float32, input.shape(), perm);
    tp->name(op.name() + "/transpose");

    input_tensors_.emplace(&tp->input(), input_name);
    output_tensors_.emplace(op.top(0), &tp->output());
}
