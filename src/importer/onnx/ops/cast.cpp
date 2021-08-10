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
#include <nncase/ir/graph.h>
#include <nncase/ir/ops/convert.h>
#include <nncase/ir/ops/slice.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Cast(const NodeProto &node)
{
    auto input = node.input()[0];
    auto output = node.output()[0];

    const auto in_type = get_datatype(input).value();
    const auto in_shape = get_shape(input);
    datatype_t out_type = dt_float32;
    if (auto out_type_info = get_datatype(output); out_type_info)
    {
        out_type = out_type_info.value();
    }
    else
    {
        out_type = get_datatype(static_cast<TensorProto_DataType>(get_attribute<int>(node, "to").value())).value();
    }

    auto ct = graph_.emplace<convert>(in_type, in_shape, out_type);
    ct->name("convert");

    link_input_tensor(&ct->input(), input);
    output_tensors_.emplace(node.output()[0], &ct->output());
}