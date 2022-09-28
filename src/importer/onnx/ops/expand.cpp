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
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/constant.h>
#include <xtensor/xadapt.hpp>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Expand(const NodeProto &node)
{
    assert(node.input().size() == 2);
    const auto &input = node.input()[0];
    const auto &output = node.output()[0];

    const auto input_type = get_datatype(input).value();
    const auto &input_shape = get_shape(input);

    auto shape_vec = get_constant_value<int64_t>(node.input()[1]);
    shape_t shape { shape_vec.begin(), shape_vec.end() };
    constant *con = nullptr;
    if (input_type == dt_int64)
    {
        auto ones = xt::ones<int64_t>(shape);
        std::vector<int64_t> ones_vec { ones.begin(), ones.end() };
        con = graph_.emplace<constant>(input_type, shape, ones_vec);
    }
    else if (input_type == dt_float32)
    {
        auto ones = xt::ones<float>(shape);
        std::vector<float> ones_vec { ones.begin(), ones.end() };
        con = graph_.emplace<constant>(input_type, shape, ones_vec);
    }
    else if (input_type == dt_uint8)
    {
        auto ones = xt::ones<uint8_t>(shape);
        std::vector<uint8_t> ones_vec { ones.begin(), ones.end() };
        con = graph_.emplace<constant>(input_type, shape, ones_vec);
    }
    auto op = graph_.emplace<binary>(binary_mul, input_type, input_shape, shape, value_range<float>::full());
    op->name(generate_name(node) + "(Expand)");

    op->input_b().connect(con->output());
    input_tensors_.emplace(&op->input_a(), input);
    output_tensors_.emplace(output, &op->output());
}
