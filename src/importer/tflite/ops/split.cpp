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
#include "../tflite_importer.h"
#include <nncase/ir/ops/split.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(SPLIT)
{
    auto &input = get_tensor(op.inputs(), 1);
    auto axis = load_array<int32_t>(get_tensor(op.inputs(), 0));
    auto &options = *op.builtin_options_as_SplitOptions();

    auto num_splits = options.num_splits();

    std::vector<shape_t> output_shapes(op.outputs()->size());
    for (size_t i = 0; i < op.outputs()->size(); i++)
    {
        auto &output_split = get_tensor(op.outputs(), i);
        auto shape = get_shape(output_split.shape());
        output_shapes[i] = shape;
    }
    std::vector<size_t> indices { size_t(num_splits) };
    auto split_node = graph_.emplace<split>(to_data_type(input.type()), get_shape(input.shape()), output_shapes, indices, axis[0], false);
    split_node->name(std::string(get_tensor(op.outputs(), 1).name()->string_view()) + "/split");

    link_input_tensor(&split_node->input(), op.inputs()->Get(1));
    for (size_t i = 0; i < op.outputs()->size(); i++)
    {
        link_output_tensor(op.outputs()->Get(i), &split_node->output_at(i));
    }
}