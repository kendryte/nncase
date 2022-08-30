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

#pragma once

#include <cstdint>
#include <nncase/importer/util.h>
#include <nncase/ir/connectors.h>
#include <nncase/ir/ir_types.h>
#include <nncase/ir/node.h>
#include <nncase/ir/ops/constant.h>
#include <onnx.pb.h>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <xtensor/xadapt.hpp>

namespace nncase::importer
{
class onnx_importer
{
public:
    onnx_importer(std::span<const std::uint8_t> model, ir::graph &graph);

    void import(const struct import_options &options, std::string &real_inlayout, std::string &real_outlayout);

private:
    typedef std::variant<
        float,
        std::int64_t,
        std::string>
        attribute_value_type;

    void convert_op(const onnx::NodeProto &node);
#define DEFINE_OPCODE(opcode) void convert_op_##opcode(const onnx::NodeProto &node);
#include "opcode.def"
#undef DEFINE_OPCODE

    void convert_unary(const onnx::NodeProto &node, const unary_op_t unary_op);
    void convert_binary(const onnx::NodeProto &node, const binary_op_t binary_op);
    void convert_op_logical(const onnx::NodeProto &node, const binary_op_t binary_op);
    void convert_op_arg(const onnx::NodeProto &node, reduce_arg_op_t op);
    void convert_op_compare(const onnx::NodeProto &node, const compare_op_t compare_op);
    void convert_op_compress(const onnx::NodeProto &node);

    template <bool global = false>
    void convert_pool(const onnx::NodeProto &node, const reduce_op_t reduce_op, const float initial_value);
    void convert_reduce(const onnx::NodeProto &node, const reduce_op_t reduce_op, const float initial_value);
    template <class Node>
    void convert_conv(const onnx::NodeProto &node);

    std::optional<onnx::ValueInfoProto> find_value_info(const std::string &value) const;
    nncase::ir::shape_t get_shape(const std::string &value) const;
    static nncase::ir::shape_t get_shape(const onnx::ValueInfoProto &value);
    static nncase::ir::shape_t get_shape(const onnx::TensorProto &value);
    std::optional<nncase::datatype_t> get_datatype(const std::string &value) const;
    static std::optional<nncase::datatype_t> get_datatype(const onnx::ValueInfoProto &value);
    static std::optional<nncase::datatype_t> get_datatype(const onnx::TensorProto &value);
    static std::optional<nncase::datatype_t> get_datatype(const onnx::TensorProto_DataType datatype);
    static std::optional<nncase::datatype_t> get_datatype(const onnx::AttributeProto_AttributeType type);
    template <typename T>
    static constexpr nncase::datatype_t get_datatype();

    static std::string to_string(const onnx::TensorProto_DataType datatype);
    static std::string to_string(const onnx::AttributeProto_AttributeType type);
    template <typename T>
    static std::optional<T> get_attribute(const onnx::NodeProto &node, const std::string &name);
    std::optional<onnx::TensorProto> get_initializer(const std::string &name) const;
    template <typename T, typename S = T>
    static std::vector<T> raw_to_vector(const onnx::TensorProto &tensor);
    template <typename T, typename S>
    static xt::xarray<T> raw_to(const onnx::TensorProto &tensor);
    template <typename T>
    static T to(const onnx::TensorProto &tensor);
    template <typename T>
    static T convert_to(const onnx::TensorProto &tensor);

    static constexpr std::size_t real_axis(const int axis, const std::size_t count) noexcept
    {
        return axis >= 0 ? axis : count + axis;
    }

    template <class T = int32_t>
    T get_positive_attr(const onnx::NodeProto &node, size_t max_size, const std::string &attr_name)
    {
        const auto axis_attr = get_attribute<T>(node, attr_name);
        return get_positive<T>(axis_attr.value(), max_size);
    }

    void add_convert(ir::input_connector &next_input, const std::string &onnx_input, datatype_t to_type);

    void input_convert_to_type(ir::input_connector &next_input, const std::string &onnx_input, datatype_t to_type);

    void link_input_tensor(ir::input_connector *conn, const std::string &onnx_v);

    void link_output_tensor(const std::string &onnx_v, ir::output_connector *conn);

    static std::vector<padding> parse_padding(const ir::axis_t &padding_value);

    template <typename T>
    std::optional<std::vector<T>> get_constant_input_data(const std::string &name) const;

    template <typename T, typename S = T,
        typename std::enable_if<(std::is_integral<T>::value && std::is_integral<S>::value) || (std::is_floating_point<T>::value && std::is_floating_point<S>::value)>::type * = nullptr>
    std::vector<T> get_constant_value(const std::string &name);

    template <typename T>
    ir::constant *emplace_constant(const std::optional<T> &v);

    template <class Cont>
    static xtl::span<const std::uint8_t> span_from(const Cont &data);

    nncase::ir::shape_t broadcast_shape(const nncase::ir::shape_t &v_shape, const nncase::ir::shape_t &input_shape) noexcept;
    std::string generate_name(const onnx::NodeProto &node) const;
    int64_t get_opset_version(std::string domain = "") const;

    ir::graph &graph_;
    onnx::ModelProto model_;
    std::unordered_map<std::string, int64_t> opset_map_;
    std::unordered_map<ir::input_connector *, std::string> input_tensors_;
    std::unordered_map<std::string, ir::output_connector *> output_tensors_;
    std::unordered_map<std::string, std::string> passthrough_connections_;
};

template <>
constexpr nncase::datatype_t onnx_importer::get_datatype<float>()
{
    return nncase::dt_float32;
}

template <>
constexpr nncase::datatype_t onnx_importer::get_datatype<std::uint8_t>()
{
    return nncase::dt_uint8;
}

template <>
constexpr nncase::datatype_t onnx_importer::get_datatype<std::int8_t>()
{
    return nncase::dt_int8;
}

template <>
constexpr nncase::datatype_t onnx_importer::get_datatype<std::int32_t>()
{
    return nncase::dt_int32;
}

template <>
constexpr nncase::datatype_t onnx_importer::get_datatype<std::int64_t>()
{
    return nncase::dt_int64;
}

template <typename T>
std::optional<std::vector<T>> onnx_importer::get_constant_input_data(const std::string &name) const
{
    const auto it { output_tensors_.find(name) };

    if (it == end(output_tensors_))
        return std::optional<std::vector<T>> {};

    if (it->second->type() != get_datatype<T>())
        return std::optional<std::vector<T>> {};

    const auto &node { it->second->owner() };

    if (node.runtime_opcode() != ir::op_constant)
        return std::optional<std::vector<T>> {};

    const auto &data { static_cast<const ir::constant &>(node).data() };
    const T *const ptr { reinterpret_cast<const T *>(data.data()) };
    const std::size_t size { data.size() / sizeof(T) };

    std::vector<T> result;
    result.reserve(size);

    std::transform(ptr, ptr + size, std::back_inserter(result),
        [](const auto &e) { return e; });

    return result;
}

template <typename T, typename S,
    typename std::enable_if<(std::is_integral<T>::value && std::is_integral<S>::value) || (std::is_floating_point<T>::value && std::is_floating_point<S>::value)>::type *>
std::vector<T> onnx_importer::get_constant_value(const std::string &name)
{
    std::vector<S> vec_storage;
    const auto &initializer = get_initializer(name);
    if (initializer)
    {
        vec_storage = to<std::vector<S>>(initializer.value());
    }
    else
    {
        const auto data = get_constant_input_data<S>(name);
        if (!data)
            throw std::runtime_error("Can't pull input data for <" + name + "> : only constant initialization is supported");

        vec_storage = data.value();
    }

    if constexpr (std::is_same_v<T, S>)
        return vec_storage;

    std::vector<T> vec_target;
    std::transform(vec_storage.begin(), vec_storage.end(), std::back_inserter(vec_target),
        [](const auto val) {
            T min = std::numeric_limits<T>::min();
            T max = std::numeric_limits<T>::max();
            if (val < min)
                return min;
            else if (val > max)
                return max;
            else
                return static_cast<T>(val);
        });

    return vec_target;
}

template <class Cont>
xtl::span<const std::uint8_t> onnx_importer::span_from(const Cont &data)
{
    return xtl::span<const std::uint8_t> {
        reinterpret_cast<const std::uint8_t *>(data.data()),
        data.size() * sizeof(typename Cont::value_type)
    };
}

template <>
ir::constant *onnx_importer::emplace_constant<onnx::TensorProto>(const std::optional<onnx::TensorProto> &v);
template <>
std::optional<float> onnx_importer::get_attribute<float>(const onnx::NodeProto &node, const std::string &name);
template <>
std::optional<std::int64_t> onnx_importer::get_attribute<std::int64_t>(const onnx::NodeProto &node, const std::string &name);
template <>
std::optional<int> onnx_importer::get_attribute<int>(const onnx::NodeProto &node, const std::string &name);
template <>
std::optional<std::string> onnx_importer::get_attribute<std::string>(const onnx::NodeProto &node, const std::string &name);
template <>
std::optional<onnx::TensorProto> onnx_importer::get_attribute<onnx::TensorProto>(const onnx::NodeProto &node, const std::string &name);
template <>
std::optional<std::vector<float>> onnx_importer::get_attribute<std::vector<float>>(const onnx::NodeProto &node, const std::string &name);
template <>
std::optional<std::vector<std::int64_t>> onnx_importer::get_attribute<std::vector<std::int64_t>>(const onnx::NodeProto &node, const std::string &name);
template <>
std::optional<std::vector<std::string>> onnx_importer::get_attribute<std::vector<std::string>>(const onnx::NodeProto &node, const std::string &name);
template <>
std::optional<ir::axis_t> onnx_importer::get_attribute<ir::axis_t>(const onnx::NodeProto &node, const std::string &name);

template <>
float onnx_importer::to<float>(const onnx::TensorProto &tensor);
template <>
std::uint8_t onnx_importer::to<std::uint8_t>(const onnx::TensorProto &tensor);
template <>
ir::axis_t onnx_importer::to<ir::axis_t>(const onnx::TensorProto &tensor);
template <>
xt::xarray<float> onnx_importer::to<xt::xarray<float>>(const onnx::TensorProto &tensor);
template <>
xt::xarray<std::uint8_t> onnx_importer::to<xt::xarray<std::uint8_t>>(const onnx::TensorProto &tensor);
template <>
xt::xarray<std::int32_t> onnx_importer::to<xt::xarray<std::int32_t>>(const onnx::TensorProto &tensor);
template <>
xt::xarray<std::int64_t> onnx_importer::to<xt::xarray<std::int64_t>>(const onnx::TensorProto &tensor);
template <>
std::vector<std::int64_t> onnx_importer::to<std::vector<std::int64_t>>(const onnx::TensorProto &tensor);
template <>
std::vector<float> onnx_importer::to<std::vector<float>>(const onnx::TensorProto &tensor);
template <>
xt::xarray<float> onnx_importer::convert_to<xt::xarray<float>>(const onnx::TensorProto &tensor);
}
