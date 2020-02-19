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
    template<> AttributeProto_AttributeType attribute_type<xtl::span<float>> { AttributeProto_AttributeType_FLOATS };
    template<> AttributeProto_AttributeType attribute_type<xtl::span<int64_t>> { AttributeProto_AttributeType_INTS };
    template<> AttributeProto_AttributeType attribute_type<axis_t> { AttributeProto_AttributeType_INTS };
    template<> AttributeProto_AttributeType attribute_type<xtl::span<string>> { AttributeProto_AttributeType_STRINGS };

    template<typename T> TensorProto_DataType tensor_type;

    template<> TensorProto_DataType tensor_type<float> { TensorProto_DataType_FLOAT };
    template<> TensorProto_DataType tensor_type<uint8_t> { TensorProto_DataType_UINT8 };
    template<> TensorProto_DataType tensor_type<int32_t> { TensorProto_DataType_INT32 };
    template<> TensorProto_DataType tensor_type<int64_t> { TensorProto_DataType_INT64 };

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

    const ValueInfoProto* find_value_info(const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> &collection, const string &value)
    {
        const auto it { find_if(collection.cbegin(), collection.cend(),
                [&value](const auto e)
                {
                    return value == e.name();
                }) };

        return it != collection.end() ? &(*it) : nullptr;
    }

    template<class Proto, template<class> class ProtobufCollection> const Proto* extract(const ProtobufCollection<Proto>& collection, const string &value)
    {
        const auto it
        {
            find_if(begin(collection), end(collection),
                [&value](const auto &e)
                {
                    return e.name() == value;
                })
        };

        return it != end(collection) ? &(*it) : nullptr;
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

const ValueInfoProto* onnx_importer::find_value_info(const string &value) const
{
    auto value_info_ptr { ::find_value_info(model_.graph().input(), value) };
    if (value_info_ptr)
        return value_info_ptr;

    value_info_ptr = ::find_value_info(model_.graph().value_info(), value);
    if (value_info_ptr)
        return value_info_ptr;

    value_info_ptr = ::find_value_info(model_.graph().output(), value);

    return value_info_ptr;
}

shape_t onnx_importer::get_shape(const string &value) const
{
    const auto value_info_ptr { find_value_info(value) };
    if (value_info_ptr)
        return get_shape(*value_info_ptr);

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

datatype_t onnx_importer::get_datatype(const AttributeProto_AttributeType type)
{
    switch (type)
    {
    case AttributeProto_AttributeType_FLOAT:
        return dt_float32;

    case AttributeProto_AttributeType_INT:
        return dt_uint8;

    default:
        throw runtime_error("ONNX data type " + to_string(type) + " is unsupported");
    }
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

string onnx_importer::to_string(const TensorProto_DataType datatype)
{
    switch (datatype)
    {
    default:
    case TensorProto_DataType_UNDEFINED:
        return "UNDEFINED";

    case TensorProto_DataType_FLOAT:
        return "FLOAT";

    case TensorProto_DataType_UINT8:
        return "UINT8";

    case TensorProto_DataType_INT8:
        return "INT8";

    case TensorProto_DataType_UINT16:
        return "UINT16";

    case TensorProto_DataType_INT16:
        return "INT16";

    case TensorProto_DataType_INT32:
        return "INT32";

    case TensorProto_DataType_INT64:
        return "INT64";

    case TensorProto_DataType_STRING:
        return "STRING";

    case TensorProto_DataType_BOOL:
        return "BOOL";

    case TensorProto_DataType_FLOAT16:
        return "FLOAT16";

    case TensorProto_DataType_DOUBLE:
        return "DOUBLE";

    case TensorProto_DataType_UINT32:
        return "UINT32";

    case TensorProto_DataType_UINT64:
        return "UINT64";

    case TensorProto_DataType_COMPLEX64:
        return "COMPLEX64";

    case TensorProto_DataType_COMPLEX128:
        return "COMPLEX128";

    case TensorProto_DataType_BFLOAT16:
        return "BFLOAT16";
    }
}

string onnx_importer::to_string(const AttributeProto_AttributeType type)
{
    switch (type)
    {
    default:
    case AttributeProto_AttributeType_UNDEFINED:
        return "UNDEFINED";

    case AttributeProto_AttributeType_FLOAT:
        return "FLOAT";

    case AttributeProto_AttributeType_INT:
        return "INT";

    case AttributeProto_AttributeType_STRING:
        return "STRING";

    case AttributeProto_AttributeType_TENSOR:
        return "TENSOR";

    case AttributeProto_AttributeType_GRAPH:
        return "GRAPH";

    case AttributeProto_AttributeType_SPARSE_TENSOR:
        return "SPARSE_TENSOR";

    case AttributeProto_AttributeType_FLOATS:
        return "FLOATS";

    case AttributeProto_AttributeType_INTS:
        return "INTS";

    case AttributeProto_AttributeType_STRINGS:
        return "STRINGS";

    case AttributeProto_AttributeType_TENSORS:
        return "TENSORS";

    case AttributeProto_AttributeType_GRAPHS:
        return "GRAPHS";

    case AttributeProto_AttributeType_SPARSE_TENSORS:
        return "SPARSE_TENSORS";
    }
}

onnx_importer::attribute_value_type onnx_importer::get_attribute(const onnx::NodeProto& node, const string &value) const
{
    const auto it
    {
        find_if(node.attribute().cbegin(), node.attribute().cend(),
            [&value](const auto &attr)
            {
                return attr.name() == value;
            })
    };

    if (it == node.attribute().cend())
        return attribute_value_type { };

    const auto &attr { *it };

    switch(attr.type())
    {
    case AttributeProto_AttributeType_FLOAT:
        return attr.f();

    case AttributeProto_AttributeType_INT:
        return attr.i();

    case AttributeProto_AttributeType_STRING:
        return attr.s();

    default:
        break;
    }

    return attribute_value_type { };
}

template<> optional<float> onnx_importer::get_attribute<float>(const onnx::NodeProto& node, const string &value)
{
    typedef float target_type;
    const auto* attr { extract(node.attribute(), value) };

    if (!attr)
        return optional<target_type> { };

    assert(attr->type() == attribute_type<target_type>);

    return attr->f();
}

template<> optional<int64_t> onnx_importer::get_attribute<int64_t>(const onnx::NodeProto& node, const string &value)
{
    typedef int64_t target_type;
    const auto* attr { extract(node.attribute(), value) };

    if (!attr)
        return optional<target_type> { };

    assert(attr->type() == attribute_type<target_type>);

    return attr->i();
}

template<> optional<int> onnx_importer::get_attribute<int>(const onnx::NodeProto& node, const string &value)
{
    typedef int target_type;
    const auto* attr { extract(node.attribute(), value) };

    if (!attr)
        return optional<target_type> { };

    assert(attr->type() == attribute_type<target_type>);

    return attr->i();
}

template<> optional<string> onnx_importer::get_attribute<string>(const onnx::NodeProto& node, const string &value)
{
    typedef string target_type;
    const auto* attr { extract(node.attribute(), value) };

    if (!attr)
        return optional<target_type> { };

    assert(attr->type() == attribute_type<target_type>);

    return attr->s();
}

template<> optional<xtl::span<const float>> onnx_importer::get_attribute<xtl::span<const float>>(const onnx::NodeProto& node, const string &value)
{
    typedef xtl::span<const float> target_type;
    const auto* attr { extract(node.attribute(), value) };

    if (!attr)
        return optional<target_type> { };

    assert(attr->type() == attribute_type<target_type>);

    return target_type { &(*attr->floats().begin()), &(*attr->floats().end()) };
}

template<> optional<xtl::span<const int64_t>> onnx_importer::get_attribute<xtl::span<const int64_t>>(const onnx::NodeProto& node, const string &value)
{
    typedef xtl::span<const int64_t> target_type;
    const auto* attr { extract(node.attribute(), value) };

    if (!attr)
        return optional<target_type> { };

    assert(attr->type() == attribute_type<target_type>);

    return target_type { &(*attr->ints().begin()), &(*attr->ints().end()) };
}

template<> optional<xtl::span<const string>> onnx_importer::get_attribute<xtl::span<const string>>(const onnx::NodeProto& node, const string &value)
{
    typedef xtl::span<const string> target_type;
    const auto* attr { extract(node.attribute(), value) };

    if (!attr)
        return optional<target_type> { };

    assert(attr->type() == attribute_type<target_type>);

    return target_type { &(*attr->strings().begin()), &(*attr->strings().end()) };
}

template<> optional<axis_t> onnx_importer::get_attribute<axis_t>(const onnx::NodeProto& node, const string &value)
{
    typedef axis_t target_type;
    const auto* attr { extract(node.attribute(), value) };

    if (!attr)
        return optional<target_type> { };

    assert(attr->type() == attribute_type<target_type>);

    return target_type { begin(attr->ints()), end(attr->ints()) };
}

const TensorProto& onnx_importer::get_initializer(const string &value) const
{
    const auto& graph { model_.graph() };
    const auto* initializer { extract(graph.initializer(), value) };

    if (!initializer)
        throw runtime_error("Can't find initializer for " + value);

    return *initializer;
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
    assert(tensor.data_type() == tensor_type<std::int32_t>);

    switch (tensor.data_type())
    {
    case TensorProto_DataType_INT32:
        assert(tensor.int32_data_size() > 0);
        return { &(*tensor.int32_data().begin()), &(*tensor.int32_data().end()) };

    case TensorProto_DataType_INT64:
    {
        assert(tensor.int64_data_size() > 0);
        axis_t result;
        transform(tensor.int64_data().begin(), tensor.int64_data().end(), back_inserter(result),
            [](const auto a) { return a; });

        return result;
    }
    default:
        throw runtime_error("Tensor can't be converted to axis");
    }
}

template<> xt::xarray<float> onnx_importer::to<xt::xarray<float>>(const onnx::TensorProto &tensor)
{
    assert(tensor.data_type() == tensor_type<float>);

    return xt::adapt(tensor.float_data().data(), tensor.float_data().size(), xt::no_ownership(), vector<int> { tensor.float_data().size() });
}

graph nncase::importer::import_onnx(xtl::span<const uint8_t> model)
{
    graph graph;
    onnx_importer(model, graph).import();
    return graph;
}
