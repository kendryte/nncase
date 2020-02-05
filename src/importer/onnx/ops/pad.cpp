/* Copyright 2020 Alexey Chernov <4ernov@gmail.com>
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

#include "../onnx_importer.h"

#include <cassert>

#include <hlir/graph.h>
#include <hlir/ops/pad.h>

using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

void onnx_importer::convert_op_Pad(const NodeProto& node)
{
    const auto &input { node.input()[0] };
    const auto &pads { node.input()[1] };
    const auto &output { node.output()[0] };

    const auto input_info_ptr { find_value_info(input) };

    if (!input_info_ptr)
        throw runtime_error("Can't find value info for " + input + " input");

    auto input_shape { get_shape(*input_info_ptr) };
    auto input_type { get_datatype(*input_info_ptr) };

    constexpr char constant_mode_caption[] { "constant" };
    string mode { constant_mode_caption };

    const auto mode_attr { get_attribute<string>(node, "mode") };
    if (mode_attr)
    {
        mode = mode_attr.value();
        if (mode != constant_mode_caption)
        {
            cout << "Warning: only 'constant' padding mode is supported by hardware, falling back to it" << endl;
        }
    }

    axis_t padding_value { to<axis_t>(get_initializer(pads)) };

    if (padding_value.size() != 4)
        throw runtime_error("Only 2D padding is supported");

    xt::svector<padding> new_paddings
    {
        { padding_value[0], padding_value[1] },
        { padding_value[2], padding_value[3] }
    };

    scalar constant { 0 };

    if (node.input().size() == 3)
    {
        const auto &constant_value { node.input()[2] };
        switch (input_type)
        {
        case dt_float32:
            constant = to<float>(get_initializer(constant_value));
            break;

        case dt_uint8:
            constant = to<uint8_t>(get_initializer(constant_value));
            break;
        }
    }

    auto op { graph_.emplace<pad>(input_type, input_shape, new_paddings, move(constant)) };

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}
