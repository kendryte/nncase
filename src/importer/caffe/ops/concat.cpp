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
#include <ir/ops/concat.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace caffe;

DEFINE_CAFFE_LOWER(Concat)
{
    std::vector<shape_t> input_shapes;
    for (int i = 0; i < op.bottom_size(); i++)
        input_shapes.push_back(output_tensors_.at(op.bottom(i))->shape());

    auto &param = op.concat_param();
    auto con = graph_.emplace<concat>(dt_float32, input_shapes, param.axis());
    con->name(op.name());

    for (int i = 0; i < op.bottom_size(); i++)
        input_tensors_.emplace(&con->input_at(i), op.bottom(i));
    output_tensors_.emplace(op.top(0), &con->output());
}
