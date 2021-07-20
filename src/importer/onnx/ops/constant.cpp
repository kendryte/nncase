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

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

template <>
constant *onnx_importer::emplace_constant<TensorProto>(const std::optional<TensorProto> &value)
{
    if (!value)
        return nullptr;

    const auto &v = value.value();
    shape_t shape = get_shape(v);
    const auto value_dt = get_datatype(v);

    TensorProto_DataType tensor_element_type { v.data_type() };

    switch (tensor_element_type)
    {
    case TensorProto_DataType_UINT8:
    {
        const auto &data = to<xt::xarray<uint8_t>>(v);
        std::vector<uint8_t> vec { data.begin(), data.end() };
        return graph_.emplace<constant>(value_dt.value(), shape, vec);
    }

    case TensorProto_DataType_FLOAT:
    {
        const auto &data = to<xt::xarray<float>>(v);
        std::vector<float> vec { data.begin(), data.end() };
        return graph_.emplace<constant>(value_dt.value(), shape, vec);
    }

    case TensorProto_DataType_INT32:
    {
        const auto &data = to<xt::xarray<int32_t>>(v);
        std::vector<int32_t> vec { data.begin(), data.end() };
        return graph_.emplace<constant>(value_dt.value(), shape, vec);
    }
    case TensorProto_DataType_INT64:
    {
        const auto &data = to<xt::xarray<int64_t>>(v);
        std::vector<int64_t> vec { data.begin(), data.end() };
        return graph_.emplace<constant>(value_dt.value(), shape, vec);
    }
    case TensorProto_DataType_UINT16:
    case TensorProto_DataType_INT16:
    {
        const auto &data = convert_to<xt::xarray<float>>(v);
        std::vector<float> vec { data.begin(), data.end() };
        return graph_.emplace<constant>(dt_float32, shape, vec);
    }

    default:
        throw std::runtime_error("Data type \"" + to_string(tensor_element_type) + "\" not supported");
    }
}

void onnx_importer::convert_op_Constant(const NodeProto &node)
{
    assert(node.input().size() == 0);
    assert(node.output().size() == 1);

    const auto &output = node.output()[0];

    ir::constant *op = nullptr;
    if (const auto value = get_attribute<TensorProto>(node, "value"))
    {
        op = emplace_constant(value);
    }
    else if (const auto value = get_attribute<float>(node, "value_float"))
    {
        op = graph_.emplace<constant>(value.value());
        op->name(generate_name(node) + "(Constant)");
    }
    else if (const auto value = get_attribute<std::vector<float>>(node, "value_floats"))
    {
        auto &&v = value.value();
        std::vector<float> vec { v.begin(), v.end() };
        shape_t shape { 1, v.size() };
        op = graph_.emplace<constant>(dt_float32, shape, vec);
        op->name(generate_name(node) + "(Constant)");
    }
    else if (const auto value = get_attribute<int>(node, "value_int"))
    {
        op = graph_.emplace<constant>(static_cast<uint8_t>(value.value()));
    }
    else if (const auto value = get_attribute<std::vector<int>>(node, "value_ints"))
    {
        auto &&v = value.value();
        std::vector<uint8_t> vec { v.begin(), v.end() };
        shape_t shape { 1, v.size() };
        op = graph_.emplace<constant>(dt_uint8, shape, vec);
        op->name(generate_name(node) + "(Constant)");
    }
    else
    {
        throw std::runtime_error("Constant field format is not supported.");
    }

    assert(op);

    output_tensors_.emplace(output, &op->output());
}
