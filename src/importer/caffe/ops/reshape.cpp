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
#include <ir/ops/reshape.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace caffe;

DEFINE_CAFFE_LOWER(Reshape)
{
    auto &input = *output_tensors_.at(op.bottom(0));
    auto &param = op.reshape_param();

    auto rp = graph_.emplace<reshape>(dt_float32, input.shape(), get_axis(param.shape()));

    input_tensors_.emplace(&rp->input(), op.bottom(0));
    output_tensors_.emplace(op.top(0), &rp->output());
}
