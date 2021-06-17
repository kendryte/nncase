/* Copyright 2020 Canaan Inc.
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
#include <nncase/ir/ops/onehot.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(ONE_HOT)
{
    auto &indices = get_tensor(op.inputs(), 0);
    [[maybe_unused]] auto &depth = get_tensor(op.inputs(), 1);
    [[maybe_unused]] auto &on_value = get_tensor(op.inputs(), 2);
    [[maybe_unused]] auto &off_value = get_tensor(op.inputs(), 3);
    auto &output = get_tensor(op.outputs(), 0);

    auto indices_shape = get_shape(indices.shape());
    auto out_shape = get_shape(output.shape());

    auto &options = *op.builtin_options_as_OneHotOptions();

    auto oh = graph_.emplace<onehot>(to_data_type(output.type()), indices_shape, out_shape, options.axis());
    oh->name(get_tensor(op.outputs(), 0).name()->string_view());

    link_input_tensor(&oh->indices(), op.inputs()->Get(0));
    link_input_tensor(&oh->depth(), op.inputs()->Get(1));
    link_input_tensor(&oh->on_value(), op.inputs()->Get(2));
    link_input_tensor(&oh->off_value(), op.inputs()->Get(3));
    link_output_tensor(op.outputs()->Get(0), &oh->output());
}
