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
#include <ir/ops/pad.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(PAD)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto paddings = load_tensor<int32_t, 2>(get_tensor(op.inputs(), 1));
    auto &options = *op.builtin_options_as_PadOptions();

    xt::svector<padding> new_paddings;
    for (size_t i = 0; i < paddings.shape()[0]; i++)
        new_paddings.push_back(padding { paddings(i, 0), paddings(i, 1) });

    auto node = graph_.emplace<pad>(dt_float32, get_shape(input.shape()), new_paddings, 0.f);

    input_tensors_.emplace(&node->input(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &node->output());
}
