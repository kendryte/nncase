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

void onnx_importer::convert_op_EyeLike(const NodeProto &node)
{
    assert(node.input().size() == 1);
    const auto &input = node.input()[0];
    const auto &output = node.output()[0];

    const auto &op_name { generate_name(node) };
    const auto input_type = get_datatype(input).value();
    const auto &input_shape = get_shape(input);
    assert(input_shape.size() == 2);

    const auto &k = get_attribute<int>(node, "k");

    //datatype_t dt = dt_float32;
    using T = float;

    std::vector<T> out_vec(input_shape[0] * input_shape[1], (T)(0));

    if (k.value() >= 0)
    {
        for (size_t i = 0; i < out_vec.size(); i++)
        {
            if ((i - k.value()) % (input_shape[1] + 1) == 0)
            {
                out_vec[i] = 1;
                if ((i + 1) % input_shape[1] == 0)
                {
                    break;
                }
            }
        }
    }
    else
    {
        for (size_t i = 0; i < out_vec.size(); i++)
        {
            if (((i - k.value()) % (input_shape[1] + 1) == 0) && (i >= (-k.value()) * input_shape[1]))
            {
                out_vec[i] = 1;
            }
        }
    }

    auto con = graph_.emplace<constant>(input_type, input_shape, out_vec);
    con->name(op_name + ".con(EyeLike)");

    auto zero = graph_.emplace<constant>(0.f);
    zero->name(op_name + ".zero(EyeLike)");

    auto mul = graph_.emplace<binary>(binary_mul, input_type, input_shape, zero->output().shape(), value_range<float>::full());
    mul->name(op_name + ".mul(EyeLike)");

    auto add = graph_.emplace<binary>(binary_add, input_type, mul->output().shape(), con->output().shape(), value_range<float>::full());
    add->name(op_name + ".add(EyeLike)");

    mul->input_b().connect(zero->output());
    add->input_a().connect(mul->output());
    add->input_b().connect(con->output());
    input_tensors_.emplace(&mul->input_a(), input);
    output_tensors_.emplace(output, &add->output());
}
