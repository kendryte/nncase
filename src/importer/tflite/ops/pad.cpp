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
#include <nncase/ir/ops/pad.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(PAD)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto paddings = load_tensor<int32_t, 2>(get_tensor(op.inputs(), 1));
    [[maybe_unused]] auto &options = *op.builtin_options_as_PadOptions();

    xt::svector<padding> new_paddings;
    for (size_t i = 0; i < paddings.shape()[0]; i++)
        new_paddings.push_back(padding { paddings(i, 0), paddings(i, 1) });

    // TODO: if zero_point is by channel, it can't be used to represent pad_value
    // pad_value will be different when input_type is different
    // f32: 0.f  uint8: static_cast<int8_t>(128)  int8: static_cast<int8_t>(0)
    auto pad_value = (input.type() == tflite::TensorType_FLOAT32) ? 0.f
                                                                  : ((input.type() == tflite::TensorType_UINT8) ? static_cast<int8_t>(128)
                                                                                                                : static_cast<int8_t>(0));

    auto node = graph_.emplace<pad>(to_data_type(input.type()), get_shape(input.shape()), new_paddings, pad_constant, pad_value);

    node->name(get_tensor(op.outputs(), 0).name()->string_view());

    link_input_tensor(&node->input(), op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), &node->output());
}

DEFINE_TFLITE_LOWER(PADV2)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto paddings = load_tensor<int32_t, 2>(get_tensor(op.inputs(), 1));
    auto pad_value = load_scalar(get_tensor(op.inputs(), 2));
    [[maybe_unused]] auto &options = *op.builtin_options_as_PadV2Options();

    xt::svector<padding> new_paddings;
    for (size_t i = 0; i < paddings.shape()[0]; i++)
        new_paddings.push_back(padding { paddings(i, 0), paddings(i, 1) });

    auto node = graph_.emplace<pad>(to_data_type(input.type()), get_shape(input.shape()), new_paddings, pad_constant, pad_value);

    node->name(get_tensor(op.outputs(), 0).name()->string_view());

    link_input_tensor(&node->input(), op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), &node->output());
}

DEFINE_TFLITE_LOWER(MIRROR_PAD)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto paddings = load_tensor<int32_t, 2>(get_tensor(op.inputs(), 1));
    auto &options = *op.builtin_options_as_MirrorPadOptions();
    auto tmp = options.mode();
    pad_mode_t mode ;
    switch(tmp)
    {
        case 0:
            mode = pad_reflect;
            break;
        case 1:
            mode = pad_symmetric;
            break;
        default:
            throw std::runtime_error("Unsupport Pad Mode!");
            break;
    }
    xt::svector<padding> new_paddings;
    for (size_t i = 0; i < paddings.shape()[0]; i++)
        new_paddings.push_back(padding { paddings(i, 0), paddings(i, 1) });

    auto node = graph_.emplace<pad>(to_data_type(input.type()), get_shape(input.shape()), new_paddings, mode, 0.f);

    node->name(get_tensor(op.outputs(), 0).name()->string_view());

    link_input_tensor(&node->input(), op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), &node->output());
}
