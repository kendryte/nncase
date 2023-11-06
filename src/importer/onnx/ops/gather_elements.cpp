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

#include "../onnx_importer.h"
#include <nncase/importer/util.h>
#include <nncase/ir/ops/convert.h>
#include <nncase/ir/ops/gather_elements.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_GatherElements(const NodeProto &node)
{
    const auto &input = node.input()[0];
    const auto &indices = node.input()[1];
    const auto &output = node.output()[0];

    const datatype_t input_type = get_datatype(input).value();
    const datatype_t indices_type = get_datatype(indices).value();
    const auto input_shape = get_shape(input);
    const auto indices_shape = get_shape(indices);
    const auto out_shape = get_shape(output);

    auto axis = get_attribute<int32_t>(node, "axis").value_or(0);
    if (axis < 0)
    {
        axis += static_cast<int32_t>(input_shape.size());
    }

    auto ga = graph_.emplace<gather_elements>(input_type, dt_int64, input_shape, indices_shape, out_shape, axis);

    auto mid_ptr = &ga->indices();
    if (indices_type == dt_int32)
    {
        auto cvt = graph_.emplace<convert>(indices_type, indices_shape, dt_int64);
        cvt->name(ga->name() + "(cvt_int_to_int64)");
        ga->indices().connect(cvt->output());
        mid_ptr = &cvt->input();
    }

    link_input_tensor(&ga->input(), input);
    link_input_tensor(mid_ptr, indices);
    link_output_tensor(output, &ga->output());
}