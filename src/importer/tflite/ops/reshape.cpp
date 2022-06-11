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
#include <nncase/ir/ops/bitcast.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(RESHAPE)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &output = get_tensor(op.outputs(), 0);

    auto node = graph_.emplace<bitcast>(to_data_type(input.type()), get_shape(input.shape()), get_shape(output.shape()));
    node->name(get_tensor(op.outputs(), 0).name()->string_view());

    link_input_tensor(&node->input(), op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), &node->output());
}

DEFINE_TFLITE_LOWER(SQUEEZE)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &options = *op.builtin_options_as_SqueezeOptions();
    auto input_shape = get_shape(input.shape());
    auto input_shape_size = input_shape.size();

    // modify the negative dim as positive
    std::vector<int32_t> squeeze_dims(options.squeeze_dims()->begin(), options.squeeze_dims()->end());
    for (size_t i = 0; i < squeeze_dims.size(); i++)
    {
        if (squeeze_dims[i] < 0)
            squeeze_dims[i] += static_cast<int32_t>(input_shape_size);
    }

    shape_t new_shape;
    for (size_t i = 0; i < input_shape_size; i++)
    {
        if (std::find(squeeze_dims.begin(), squeeze_dims.end(), (int32_t)i) == squeeze_dims.end())
            new_shape.push_back(input.shape()->Get(i));
    }

    auto node = graph_.emplace<bitcast>(to_data_type(input.type()), input_shape, new_shape);
    node->name(get_tensor(op.outputs(), 0).name()->string_view());

    link_input_tensor(&node->input(), op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), &node->output());
}

DEFINE_TFLITE_LOWER(EXPAND_DIMS)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &output = get_tensor(op.outputs(), 0);

    auto node = graph_.emplace<bitcast>(to_data_type(input.type()), get_shape(input.shape()), get_shape(output.shape()));
    node->name(get_tensor(op.outputs(), 0).name()->string_view());

    link_input_tensor(&node->input(), op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), &node->output());
}
