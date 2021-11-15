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
#include <cassert>
#include <nncase/ir/graph.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/transpose.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_SpaceToDepth(const NodeProto &node)
{
    const auto &input = node.input()[0];
    const auto input_type = get_datatype(input).value();
    auto input_shape = get_shape(input);
    assert(input_shape.size() == 4);
    const auto &output = node.output()[0];
    const auto &op_name { generate_name(node) };

    // blocksize
    auto blocksize_attr = get_attribute<int>(node, "blocksize");
    assert(blocksize_attr);
    auto blocksize = static_cast<size_t>(blocksize_attr.value());

    // reshape_1
    shape_t new_shape { input_shape[0], input_shape[1], input_shape[2] / blocksize, blocksize, input_shape[3] / blocksize, blocksize };
    auto bc1 = graph_.emplace<bitcast>(input_type, input_shape, new_shape);
    bc1->name(op_name + "(Reshape_1)");

    // transpose
    axis_t perm = { 0, 3, 5, 1, 2, 4 };
    auto tp = graph_.emplace<transpose>(bc1->output().type(), bc1->output().shape(), perm);
    tp->name(op_name + "(Transpose)");

    // reshape_2
    new_shape.clear();
    new_shape.assign({ input_shape[0], input_shape[1] * blocksize * blocksize, input_shape[2] / blocksize, input_shape[3] / blocksize });
    auto bc2 = graph_.emplace<bitcast>(tp->output().type(), tp->output().shape(), new_shape);
    bc2->name(op_name + "(Reshape_2)");

    tp->input().connect(bc1->output());
    bc2->input().connect(tp->output());
    input_tensors_.emplace(&bc1->input(), input);
    output_tensors_.emplace(output, &bc2->output());
}
