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

#include "onnx_importer.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/message.h>
#include <importer/importer.h>
#include <hlir/graph.h>

using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

namespace
{
    template<typename T> AttributeProto_AttributeType attribute_type;

    template<> AttributeProto_AttributeType attribute_type<float> { AttributeProto_AttributeType_FLOAT };
    template<> AttributeProto_AttributeType attribute_type<int64_t> { AttributeProto_AttributeType_INT };
    template<> AttributeProto_AttributeType attribute_type<int> { AttributeProto_AttributeType_INT };
    template<> AttributeProto_AttributeType attribute_type<string> { AttributeProto_AttributeType_STRING };
    template<> AttributeProto_AttributeType attribute_type<TensorProto> { AttributeProto_AttributeType_TENSOR };
    template<> AttributeProto_AttributeType attribute_type<vector<float>> { AttributeProto_AttributeType_FLOATS };
    template<> AttributeProto_AttributeType attribute_type<vector<int>> { AttributeProto_AttributeType_INTS };
    template<> AttributeProto_AttributeType attribute_type<axis_t> { AttributeProto_AttributeType_INTS };
    template<> AttributeProto_AttributeType attribute_type<vector<string>> { AttributeProto_AttributeType_STRINGS };

    template<typename T> TensorProto_DataType tensor_type;

    template<> TensorProto_DataType tensor_type<float> { TensorProto_DataType_FLOAT };
    template<> TensorProto_DataType tensor_type<uint8_t> { TensorProto_DataType_UINT8 };
    template<> TensorProto_DataType tensor_type<int32_t> { TensorProto_DataType_INT32 };
    template<> TensorProto_DataType tensor_type<int64_t> { TensorProto_DataType_INT64 };

    constexpr bool native_little_endian { !static_cast<bool>(NATIVE_IS_BIG_ENDIAN) };

    template <typename Proto>
    bool ParseProtoFromBytes(Proto* proto, const unsigned char* buffer, size_t length)
    {
        // Total bytes hard limit / warning limit are set to 1GB and 512MB
        // respectively.
        ::google::protobuf::io::ArrayInputStream input_stream(buffer, static_cast<int>(length));
        ::google::protobuf::io::CodedInputStream coded_stream(&input_stream);
        coded_stream.SetTotalBytesLimit((2048LL << 20) - 1, 512LL << 20);
        return proto->ParseFromCodedStream(&coded_stream);
    }

    template<class Proto, template<class> class ProtobufCollection> optional<Proto> extract(const ProtobufCollection<Proto>& collection, const string &value)
    {
        const auto it
        {
            find_if(begin(collection), end(collection),
                [&value](const auto &e)
                {
                    return e.name() == value;
                })
        };

        return it != end(collection) ? *it : optional<Proto> { };
    }

    template<typename T> T le_to_native(const unsigned char* data);

    template<> float le_to_native(const unsigned char* data)
    {
        uint32_t result
        {
            uint32_t(data[0] << 0) |
            uint32_t(data[1] << 8) |
            uint32_t(data[2] << 16) |
            uint32_t(data[3] << 24)
        };

        return *reinterpret_cast<const float*>(&result);
    }
}

onnx_importer::onnx_importer(xtl::span<const uint8_t> model, hlir::graph &graph)
    : graph_(graph)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if (!ParseProtoFromBytes(&model_, model.data(), model.size()))
        throw std::runtime_error("Invalid ONNX model");
}

void onnx_importer::import()
{
    const auto& graph { model_.graph() };

    for (const auto& node : graph.node())
        convert_op(node);

    // create inputs
    for (const auto& input_info : graph.input())
    {
        const auto& input_name { input_info.name() };
        auto&& input_shape { get_shape(input_info) };
        const auto input_dt { get_datatype(input_info) };

        if (!input_dt)
            throw runtime_error("Data type of input \"" + input_name + "\" is not supported");

        auto node { graph_.emplace<input_node>(input_dt.value(), input_shape) };
        node->name(input_name);

        output_tensors_.emplace(input_name, &node->output());
    }

    // create outputs
    for (const auto& output_info : graph.output())
    {
        const auto& output_name { output_info.name() };
        auto&& output_shape { get_shape(output_info) };
        const auto output_dt { get_datatype(output_info) };

        if (!output_dt)
            throw runtime_error("Data type of output \"" + output_name + "\" is not supported");

        auto node { graph_.emplace<output_node>(output_dt.value(), output_shape) };
        node->name(output_name);

        input_tensors_.emplace(&node->input(), output_name);
    }

    // connect tensors
    for (auto &&in : input_tensors_)
    {
        auto out_it = output_tensors_.find(in.second);
        if (out_it != output_tensors_.end())
        {
            in.first->connect(*out_it->second);
        }
        else
        {
            throw runtime_error("Cannot find associated output node for input " + string(in.second));
        }
    }
}

void onnx_importer::convert_op(const NodeProto &node)
{
    auto op_type = node.op_type();

#define DEFINE_OPCODE(opcode)                                \
    if (op_type == #opcode)                                  \
        return convert_op_##opcode(node);
    #include "opcode.def"
#undef DEFINE_OPCODE

    throw runtime_error("Not supported ONNX opcode: " + op_type);
}

optional<ValueInfoProto> onnx_importer::find_value_info(const string &value) const
{
    auto value_info { extract(model_.graph().input(), value) };
    if (value_info)
        return value_info;

    value_info = extract(model_.graph().value_info(), value);
    if (value_info)
        return value_info;

    value_info = extract(model_.graph().output(), value);

    return value_info;
}

shape_t onnx_importer::get_shape(const string &value) const
{
    const auto oit { output_tensors_.find(value) };
    if (oit != end(output_tensors_))
    {
        return oit->second->shape();
    }

    const auto value_info { find_value_info(value) };
    if (value_info)
        return get_shape(value_info.value());

    const auto initializer { get_initializer(value) };
    if (initializer)
	    return get_shape(initializer.value());

    throw runtime_error("Can't find value info for " + value + " to parse its shape");
}

shape_t onnx_importer::get_shape(const ValueInfoProto &value_info)
{
    const auto& type { value_info.type() };
    assert(type.value_case() == TypeProto::kTensorType);

    const auto& shape { type.tensor_type().shape() };

    shape_t result_shape;
    for (const auto& dim : shape.dim())
    {
        switch (dim.value_case())
        {
        case TensorShapeProto_Dimension::kDimValue:
            result_shape.push_back(dim.dim_value());
            break;

        case TensorShapeProto_Dimension::kDimParam:
            result_shape.push_back(-1);
            break;
        }
    }

    return result_shape;
}

shape_t onnx_importer::get_shape(const TensorProto &value)
{
    const auto& shape { value.dims() };

    shape_t result_shape { begin(shape), end(shape) };

    return result_shape;
}

optional<datatype_t> onnx_importer::get_datatype(const string &value) const
{
    const auto oit { output_tensors_.find(value) };
    if (oit != end(output_tensors_))
    {
        return oit->second->type();
    }

    const auto value_info { find_value_info(value) };
    if (value_info)
        return get_datatype(value_info.value());

    const auto initializer { get_initializer(value) };
    if (initializer)
	    return get_datatype(initializer.value());

    return optional<datatype_t> { };
}


optional<datatype_t> onnx_importer::get_datatype(const ValueInfoProto &value_info)
{
    const auto& type { value_info.type() };
    assert(type.value_case() == TypeProto::kTensorType);

    return get_datatype(static_cast<TensorProto_DataType>(type.tensor_type().elem_type()));
}

optional<datatype_t> onnx_importer::get_datatype(const TensorProto &value)
{
    const auto& type { value.data_type() };

    return get_datatype(static_cast<TensorProto_DataType>(type));
}

optional<datatype_t> onnx_importer::get_datatype(const TensorProto_DataType datatype)
{
    switch (datatype)
    {
    case TensorProto_DataType_FLOAT:
        return dt_float32;

    case TensorProto_DataType_UINT8:
        return dt_uint8;

    default:
        return optional<datatype_t> { };
    }
}

optional<datatype_t> onnx_importer::get_datatype(const AttributeProto_AttributeType type)
{
    switch (type)
    {
    case AttributeProto_AttributeType_FLOAT:
        return dt_float32;

    case AttributeProto_AttributeType_INT:
        return dt_uint8;

    default:
	    return optional<datatype_t> { };
    }
}

string onnx_importer::to_string(const TensorProto_DataType datatype)
{
    switch (datatype)
    {
    default:
    case TensorProto_DataType_UNDEFINED:
        return "UNDEFINED"s;

    case TensorProto_DataType_FLOAT:
        return "FLOAT"s;

    case TensorProto_DataType_UINT8:
        return "UINT8"s;

    case TensorProto_DataType_INT8:
        return "INT8"s;

    case TensorProto_DataType_UINT16:
        return "UINT16"s;

    case TensorProto_DataType_INT16:
        return "INT16"s;

    case TensorProto_DataType_INT32:
        return "INT32"s;

    case TensorProto_DataType_INT64:
        return "INT64"s;

    case TensorProto_DataType_STRING:
        return "STRING"s;

    case TensorProto_DataType_BOOL:
        return "BOOL"s;

    case TensorProto_DataType_FLOAT16:
        return "FLOAT16"s;

    case TensorProto_DataType_DOUBLE:
        return "DOUBLE"s;

    case TensorProto_DataType_UINT32:
        return "UINT32"s;

    case TensorProto_DataType_UINT64:
        return "UINT64"s;

    case TensorProto_DataType_COMPLEX64:
        return "COMPLEX64"s;

    case TensorProto_DataType_COMPLEX128:
        return "COMPLEX128"s;

    case TensorProto_DataType_BFLOAT16:
        return "BFLOAT16"s;
    }
}

string onnx_importer::to_string(const AttributeProto_AttributeType type)
{
    switch (type)
    {
    default:
    case AttributeProto_AttributeType_UNDEFINED:
        return "UNDEFINED"s;

    case AttributeProto_AttributeType_FLOAT:
        return "FLOAT"s;

    case AttributeProto_AttributeType_INT:
        return "INT"s;

    case AttributeProto_AttributeType_STRING:
        return "STRING"s;

    case AttributeProto_AttributeType_TENSOR:
        return "TENSOR"s;

    case AttributeProto_AttributeType_GRAPH:
        return "GRAPH"s;

    case AttributeProto_AttributeType_SPARSE_TENSOR:
        return "SPARSE_TENSOR"s;

    case AttributeProto_AttributeType_FLOATS:
        return "FLOATS"s;

    case AttributeProto_AttributeType_INTS:
        return "INTS"s;

    case AttributeProto_AttributeType_STRINGS:
        return "STRINGS"s;

    case AttributeProto_AttributeType_TENSORS:
        return "TENSORS"s;

    case AttributeProto_AttributeType_GRAPHS:
        return "GRAPHS"s;

    case AttributeProto_AttributeType_SPARSE_TENSORS:
        return "SPARSE_TENSORS"s;
    }
}

template<> optional<float> onnx_importer::get_attribute<float>(const onnx::NodeProto& node, const string &value)
{
    typedef float target_type;
    const auto& attr { extract(node.attribute(), value) };

    if (!attr)
        return optional<target_type> { };

    assert(attr.value().type() == attribute_type<target_type>);

    return attr.value().f();
}

template<> optional<int64_t> onnx_importer::get_attribute<int64_t>(const onnx::NodeProto& node, const string &value)
{
    typedef int64_t target_type;
    const auto& attr { extract(node.attribute(), value) };

    if (!attr)
        return optional<target_type> { };

    assert(attr.value().type() == attribute_type<target_type>);

    return attr.value().i();
}

template<> optional<int> onnx_importer::get_attribute<int>(const onnx::NodeProto& node, const string &value)
{
    typedef int target_type;
    const auto& attr { extract(node.attribute(), value) };

    if (!attr)
        return optional<target_type> { };

    assert(attr.value().type() == attribute_type<target_type>);

    return attr.value().i();
}

template<> optional<string> onnx_importer::get_attribute<string>(const onnx::NodeProto& node, const string &value)
{
    typedef string target_type;
    const auto& attr { extract(node.attribute(), value) };

    if (!attr)
        return optional<target_type> { };

    assert(attr.value().type() == attribute_type<target_type>);

    return attr.value().s();
}

template<> optional<TensorProto> onnx_importer::get_attribute<TensorProto>(const onnx::NodeProto& node, const string &value)
{
    typedef TensorProto target_type;
    const auto& attr { extract(node.attribute(), value) };

    if (!attr)
        return optional<target_type> { };

    assert(attr.value().type() == attribute_type<target_type>);

    return attr.value().t();
}

template<> optional<vector<float>> onnx_importer::get_attribute<vector<float>>(const onnx::NodeProto& node, const string &value)
{
    typedef vector<float> target_type;
    const auto& attr { extract(node.attribute(), value) };

    if (!attr)
        return optional<target_type> { };

    assert(attr.value().type() == attribute_type<target_type>);

    return target_type { begin(attr.value().floats()), end(attr.value().floats()) };
}

template<> optional<vector<int>> onnx_importer::get_attribute<vector<int>>(const onnx::NodeProto& node, const string &value)
{
    typedef vector<int> target_type;
    const auto& attr { extract(node.attribute(), value) };

    if (!attr)
        return optional<target_type> { };

    assert(attr.value().type() == attribute_type<target_type>);

    return target_type { begin(attr.value().ints()), end(attr.value().ints()) };
}

template<> optional<vector<string>> onnx_importer::get_attribute<vector<string>>(const onnx::NodeProto& node, const string &value)
{
    typedef vector<string> target_type;
    const auto& attr { extract(node.attribute(), value) };

    if (!attr)
        return optional<target_type> { };

    assert(attr.value().type() == attribute_type<target_type>);

    return target_type { begin(attr.value().strings()), end(attr.value().strings()) };
}

template<> optional<axis_t> onnx_importer::get_attribute<axis_t>(const onnx::NodeProto& node, const string &value)
{
	const auto& extracted { get_attribute<vector<int>>(node, value) };

	if (!extracted)
		return optional<axis_t> { };

	axis_t result { begin(extracted.value()), end(extracted.value()) };

	return result;
}

optional<TensorProto> onnx_importer::get_initializer(const string &value) const
{
    const auto& graph { model_.graph() };
    const auto& initializer { extract(graph.initializer(), value) };

    return initializer;
}

template<typename T, typename S = T> vector<T> onnx_importer::raw_to_vector(const onnx::TensorProto &tensor)
{
    typedef T target_type;
    typedef S storage_type;

    const storage_type* const ptr { reinterpret_cast<const storage_type*>(tensor.raw_data().data()) };
    const size_t size { tensor.raw_data().size() / sizeof(storage_type) };

    if constexpr (native_little_endian)
    {
        return vector<target_type>{ ptr, ptr + size };
    }
    else
    {
        vector<target_type> data;
        data.reserve(size);
        transform(ptr, ptr + size, back_inserter(data),
            [](const auto& e)
            {
                return le_to_native<storage_type>(reinterpret_cast<const byte*>(&e));
            });

        return data;
    }
}

template<typename T, typename S> xt::xarray<T> onnx_importer::raw_to(const onnx::TensorProto &tensor)
{
    return xt::adapt(raw_to_vector<T, S>(tensor), get_shape(tensor));
}

template<> float onnx_importer::to<float>(const onnx::TensorProto &tensor)
{
    assert(tensor.data_type() == tensor_type<float>);
    assert(tensor.float_data_size() > 0);

    return tensor.float_data()[0];
}

template<> uint8_t onnx_importer::to<uint8_t>(const onnx::TensorProto &tensor)
{
    assert(tensor.data_type() == tensor_type<uint8_t>);
    assert(tensor.uint64_data_size() > 0);

    return static_cast<uint8_t>(tensor.uint64_data()[0]);
}

template<> axis_t onnx_importer::to<axis_t>(const onnx::TensorProto &tensor)
{
	assert(tensor.data_type() == tensor_type<std::uint8_t> ||
		tensor.data_type() == tensor_type<std::int8_t> ||
		tensor.data_type() == tensor_type<std::int16_t> ||
		tensor.data_type() == tensor_type<std::uint16_t> ||
		tensor.data_type() == tensor_type<std::int32_t> ||
		tensor.data_type() == tensor_type<std::int64_t>);

    if (!tensor.int32_data().empty())
    {
        axis_t result { begin(tensor.int32_data()), end(tensor.int32_data()) };
        return result;
    }

    if (!tensor.int64_data().empty())
    {
        axis_t result { begin(tensor.int64_data()), end(tensor.int64_data()) };
        return result;
    }

    xt::xarray<int> content;
    switch (tensor.data_type())
    {
    case TensorProto_DataType_INT32:
        content = raw_to<int, int32_t>(tensor);
        break;

    case TensorProto_DataType_INT16:
        content = raw_to<int, int16_t>(tensor);
        break;

    case TensorProto_DataType_INT8:
        content = raw_to<int, int8_t>(tensor);
        break;

    case TensorProto_DataType_UINT16:
        content = raw_to<int, uint16_t>(tensor);
        break;

    case TensorProto_DataType_UINT8:
        content = raw_to<int, uint8_t>(tensor);
        break;

    case TensorProto_DataType_INT64:
        content = raw_to<int, int64_t>(tensor);
        break;

    default:
        throw runtime_error("Tensor can't be converted to axis");
    }

    return axis_t { begin(content), end(content) };

}

template<> xt::xarray<float> onnx_importer::to<xt::xarray<float>>(const onnx::TensorProto &tensor)
{
    assert(tensor.data_type() == tensor_type<float>);

    if (!tensor.float_data().empty())
    {
	    return xt::adapt(vector<float> { begin(tensor.float_data()), end(tensor.float_data()) }, get_shape(tensor));
    }
    else
    {
	    return raw_to<float, float>(tensor);
    }
}

template<> xt::xarray<uint8_t> onnx_importer::to<xt::xarray<uint8_t>>(const onnx::TensorProto &tensor)
{
    assert(tensor.data_type() == tensor_type<uint8_t>);

    if (!tensor.int32_data().empty())
    {
	    return xt::adapt(vector<uint8_t> { begin(tensor.int32_data()), end(tensor.int32_data()) }, get_shape(tensor));
    }
    else
    {
	    typedef uint8_t target_type;
        const target_type* const ptr { reinterpret_cast<const target_type*>(tensor.raw_data().data()) };
        const size_t size { tensor.raw_data().size() / sizeof(target_type) };
        return xt::adapt(vector<target_type> { ptr, ptr + size }, get_shape(tensor));
    }
}

template<> xt::xarray<float> onnx_importer::convert_to<xt::xarray<float>>(const onnx::TensorProto &tensor)
{
    if (tensor.data_type() == TensorProto_DataType_FLOAT)
        return to<xt::xarray<float>>(tensor);

    if (!tensor.int32_data().empty())
    {
	    return xt::adapt(vector<float> {begin(tensor.int32_data()), end(tensor.int32_data()) }, get_shape(tensor));
    }

    if (!tensor.int64_data().empty())
    {
	    return xt::adapt(vector<float> {begin(tensor.int64_data()), end(tensor.int64_data()) }, get_shape(tensor));
    }

    switch (tensor.data_type())
    {
    case TensorProto_DataType_INT32:
        return raw_to<float, int32_t>(tensor);

    case TensorProto_DataType_INT16:
        return raw_to<float, int16_t>(tensor);

    case TensorProto_DataType_INT8:
        return raw_to<float, int8_t>(tensor);

    case TensorProto_DataType_UINT16:
        return raw_to<float, uint16_t>(tensor);

    case TensorProto_DataType_UINT8:
        return raw_to<float, uint8_t>(tensor);

    case TensorProto_DataType_BOOL:
        return raw_to<float, bool>(tensor);

    case TensorProto_DataType_INT64:
        return raw_to<float, int64_t>(tensor);
    }

    throw runtime_error("Unsupported conversion from " + to_string(static_cast<TensorProto_DataType>(tensor.data_type())) + " to float tensor");
}

std::vector<padding> onnx_importer::parse_padding(const axis_t& padding_value)
{
    std::vector<padding> result;

    assert(!(padding_value.size() % 2));

    const size_t middle { padding_value.size() / 2 };
    for (size_t i = middle - 2; i < middle; ++i)
        result.push_back(padding { padding_value[i], padding_value[i + middle] });

    return result;
}

graph nncase::importer::import_onnx(xtl::span<const uint8_t> model)
{
    graph graph;
    onnx_importer(model, graph).import();
    return graph;
}
