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

template <typename T>
constant *generate_constant(ir::graph &g, xt::xarray<T> &src, shape_t shape)
{
    auto dst = xt::broadcast(src, shape);
    std::vector<T> vec { dst.begin(), dst.end() };
    auto op = g.emplace<constant>(to_datatype<T>(), shape, vec);
    return op;
}

void onnx_importer::convert_op_ConstantOfShape(const NodeProto &node)
{
    const auto &input = node.input()[0];
    const auto &output = node.output()[0];

    // input(shape)
    shape_t shape;
    auto shape_vec = get_constant_value<int64_t>(input);
    if (std::all_of(shape_vec.begin(), shape_vec.end(), [](int64_t i) { return i == 0; }))
    {
        shape.assign({ 1 });
    }
    else
    {
        shape.assign(shape_vec.begin(), shape_vec.end());
    }

    // value attribute(type + value)
    constant *op = nullptr;
    datatype_t type = dt_float32;
    auto value_attr = get_attribute<TensorProto>(node, "value");
    if (!value_attr)
    {
        xt::xarray<float> src = { 0.f };
        op = generate_constant(graph_, src, shape);
    }
    else
    {
        auto tensor = value_attr.value();
        if (tensor.has_data_type())
            type = get_datatype(static_cast<TensorProto_DataType>(tensor.data_type())).value();

        switch (type)
        {
        case dt_float32:
        {
            auto src = to<xt::xarray<float>>(tensor);
            op = generate_constant(graph_, src, shape);
            break;
        }
        default:
        {
            std::cerr << "unsupported type: " << type << std::endl;
            std::abort();
            break;
        }
        }
    }

    assert(op);
    op->name(generate_name(node) + ".const(ConstantOfShape)");
    output_tensors_.emplace(output, &op->output());
}