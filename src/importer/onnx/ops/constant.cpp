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
#include <hlir/ops/constant.h>


using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

void onnx_importer::convert_op_Constant(const NodeProto &node)
{
    assert(node.input().size() == 0);
    assert(node.output().size() == 1);

    const auto &output { node.output()[0] };

    hlir::constant* op { };
    if (const auto value { get_attribute<TensorProto>(node, "value") })
    {
        const auto& v { value.value() };
        shape_t shape { get_shape(v) };
        const auto value_dt { get_datatype(v) };

        TensorProto_DataType tensor_element_type { v.data_type() };

        switch (tensor_element_type)
        {
        case TensorProto_DataType_UINT8:
        {
            const auto& data { to<xt::xarray<uint8_t>>(v) };
            xtl::span<const uint8_t> vec { data };
            op = graph_.emplace<constant>(value_dt.value(), move(shape), vec);
            break;
        }

        case TensorProto_DataType_FLOAT:
        {
            const auto& data { to<xt::xarray<float>>(v) };
            op = graph_.emplace<constant>(value_dt.value(), move(shape), span_from(data));
            break;
        }

        case TensorProto_DataType_UINT16:
        case TensorProto_DataType_INT16:
        case TensorProto_DataType_INT32:
        case TensorProto_DataType_INT64:
        {
            if (tensor_element_type == TensorProto_DataType_INT32 || tensor_element_type == TensorProto_DataType_INT64)
            {
                cout << "Constants of types int32 and int64 are represented as float32 and may suffer rounding errors if mantissa width is exceeded" << endl;
            }

            const auto& data { convert_to<xt::xarray<float>>(v) };
            op = graph_.emplace<constant>(dt_float32, move(shape), span_from(data));
            break;
        }

        default:
            throw runtime_error("Data type \"" +  to_string(tensor_element_type) + "\" not supported");
        }
    }
    else if (const auto value { get_attribute<float>(node, "value_float") })
    {
        op = graph_.emplace<constant>(value.value());
    }
    else if (const auto value { get_attribute<int>(node, "value_int") })
    {
        op = graph_.emplace<constant>(static_cast<uint8_t>(value.value()));
    }
    else if (const auto value { get_attribute<vector<int>>(node, "value_ints") })
    {
        auto&& v { value.value() };
        vector<uint8_t>&& vec { begin(v), end(v) };
        shape_t shape { 1, v.size() };
        op = graph_.emplace<constant>(dt_uint8, move(shape), vec);
    }
    else if (const auto value { get_attribute<vector<float>>(node, "value_floats") })
    {
        auto&& v { value.value() };
        shape_t shape { 1, v.size() };
        op = graph_.emplace<constant>(dt_float32, move(shape), span_from(v));
    }
    else
    {
        throw runtime_error("Constant field format is not supported.");
    }

    assert(op);

    output_tensors_.emplace(output, &op->output());
}
