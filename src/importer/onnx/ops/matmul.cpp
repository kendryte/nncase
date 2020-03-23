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
#include <hlir/ops/matmul.h>


using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

void onnx_importer::convert_op_MatMul(const NodeProto& node)
{
    const auto &input_a { node.input()[0] };
    const auto &input_b { node.input()[1] };
    const auto &output { node.output()[0] };

    auto&& input_a_shape = get_shape(input_a);
    auto&& input_b_shape = get_shape(input_b);

    auto&& bias(xt::zeros<float>({ input_b_shape.back() }));

    auto mmul { graph_.emplace<matmul>(move(input_a_shape), move(input_b_shape), move(bias), value_range<float>::full()) };

    input_tensors_.emplace(&mmul->input_a(), input_a);
    input_tensors_.emplace(&mmul->input_b(), input_b);
    output_tensors_.emplace(output, &mmul->output());
}
