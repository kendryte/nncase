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

#include <limits>
#include <algorithm>
#include <cassert>

#include <hlir/graph.h>
#include <hlir/ops/transpose.h>

using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

void onnx_importer::convert_op_Transpose(const NodeProto& node)
{
    const auto &input { node.input()[0] };
    const auto &output { node.output()[0] };

    const auto input_info_ptr { find_value_info(input) };

    if (!input_info_ptr)
        throw runtime_error("Can't find value info for " + input + " input");

    auto input_shape { get_shape(*input_info_ptr) };
    auto input_type { get_datatype(*input_info_ptr) };

    axis_t perm(input_shape.size());
    std::iota(begin(perm), end(perm), 0);
    std::reverse(begin(perm), end(perm));

    const auto &perm_attr { get_attribute<axis_t>(node, "perm") };
    if (perm_attr)
        perm = perm_attr.value();

    auto op { graph_.emplace<transpose>(input_type, move(input_shape), move(perm)) };

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}
