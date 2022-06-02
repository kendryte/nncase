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

void onnx_importer::convert_op_CastLike(const NodeProto &node)
{
    auto input_a = node.input()[0];
    auto input_b = node.input()[1];
    auto output = node.output()[0];

    const auto in_a_type = get_datatype(input_a).value();
    const auto in_a_shape = get_shape(input_a);
    const auto in_b_type = get_datatype(input_b).value();
    datatype_t out_type = in_b_type;

    auto cvt = graph_.emplace<convert>(in_a_type, in_a_shape, out_type);
    cvt->name("convert");

    link_input_tensor(&cvt->input(), input_a);
    output_tensors_.emplace(output, &cvt->output());
}