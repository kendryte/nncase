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
#include "../caffe_importer.h"
#include <ir/ops/binary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace caffe;

DEFINE_CAFFE_LOWER(Eltwise)
{
    auto &input_a = *output_tensors_.at(op.bottom(0));
    auto &input_b = *output_tensors_.at(op.bottom(1));
    auto &param = op.eltwise_param();

    auto add = graph_.emplace<binary>(binary_add, input_a.shape(), input_b.shape(), value_range<float>::full());

    input_tensors_.emplace(&add->input_a(), op.bottom(0));
    input_tensors_.emplace(&add->input_b(), op.bottom(1));
    output_tensors_.emplace(op.top(0), &add->output());
}
