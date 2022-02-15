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

    // op version 1 is not supported by onnx runtime, now we only support op version 6/9/13
    auto to_attr = get_attribute<int>(node, "to");
    assert(to_attr);

    out_type = get_datatype(static_cast<TensorProto_DataType>(to_attr.value())).value();

    auto cvt = graph_.emplace<convert>(in_type, in_shape, out_type);
    cvt->name("convert");

    link_input_tensor(&cvt->input(), input);
    output_tensors_.emplace(output, &cvt->output());
}