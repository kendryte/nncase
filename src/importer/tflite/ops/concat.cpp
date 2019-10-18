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
#include <ir/ops/concat.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(CONCATENATION)
{
    std::vector<shape_t> inputs_shape;
    auto &options = *op.builtin_options_as_ConcatenationOptions();

    for (auto &&in : *op.inputs())
    {
        auto &tensor = *subgraph_->tensors()->Get(in);
        inputs_shape.emplace_back(get_shape(tensor.shape()));
    }

    auto con = graph_.emplace<concat>(dt_float32, inputs_shape, options.axis());

    for (size_t i = 0; i < op.inputs()->size(); i++)
        input_tensors_.emplace(&con->input_at(i), op.inputs()->Get(i));

    output_tensors_.emplace(op.outputs()->Get(0), &con->output());
}
