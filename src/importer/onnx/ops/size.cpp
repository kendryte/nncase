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
#include <nncase/ir/graph.h>
#include <nncase/ir/ops/constant.h>
#include <xtensor/xadapt.hpp>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Size(const NodeProto &node)
{
    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    auto input_shape = get_shape(input);

    int64_t prod = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
    auto op = graph_.emplace<constant>(prod);
    op->name(generate_name(node) + ".const(Size)");
    output_tensors_.emplace(output, &op->output());
}
