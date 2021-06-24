/* Copyright 2019-2020 Canaan Inc.
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

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace caffe;

DEFINE_CAFFE_LOWER(Eltwise)
{
    // check if there are bn/scale/relu above
    std::string input_name_a = get_real_input_names(op)[0];
    std::string input_name_b = get_real_input_names(op)[1];

    auto &input_a = *output_tensors_.at(input_name_a);
    auto &input_b = *output_tensors_.at(input_name_b);
    [[maybe_unused]]auto &param = op.eltwise_param();

    auto add = graph_.emplace<binary>(binary_add, input_a.shape(), input_b.shape(), value_range<float>::full());
    add->name(op.name() + "/add");

    input_tensors_.emplace(&add->input_a(), input_name_a);
    input_tensors_.emplace(&add->input_b(), input_name_b);
    output_tensors_.emplace(op.top(0), &add->output());
}
