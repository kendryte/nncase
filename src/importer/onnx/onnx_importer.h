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

        void import(const struct import_options &options);

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

        template <bool global = false>
        void convert_pool(const onnx::NodeProto &node, const reduce_op_t reduce_op, const float initial_value);
        void convert_reduce(const onnx::NodeProto &node, const reduce_op_t reduce_op, const float initial_value);
        template <class Node>
        void convert_conv(const onnx::NodeProto &node);

        template <class Node>
        Node *add_conv_node(const onnx::NodeProto &node, ir::graph &graph, ir::shape_t &&input_shape, xt::xarray<float> &&weight_value, xt::xarray<float> &&bias_value, const std::size_t group, const std::array<padding, 2> &pads, const std::array<size_t, 2> &strides, const std::array<size_t, 2> &dilations);

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
        template <typename T, typename S>
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

        static std::vector<padding> parse_padding(const ir::axis_t &padding_value);

        template <typename T>
        std::optional<std::vector<T>> get_constant_input_data(const std::string &name) const;
        // template <typename T>
        // ir::constant *emplace_constant(const std::optional<T> &v);

        // template <class Cont>
        // static xtl::span<const std::uint8_t> span_from(const Cont &data);

        ir::graph &graph_;
        onnx::ModelProto model_;

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
    constexpr nncase::datatype_t onnx_importer::get_datatype<std::int64_t>()
    {
        return nncase::dt_uint8;
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

    // template <class Cont>
    // xtl::span<const std::uint8_t> onnx_importer::span_from(const Cont &data)
    // {
    //     return xtl::span<const std::uint8_t> {
    //         reinterpret_cast<const std::uint8_t *>(data.data()),
    //         data.size() * sizeof(typename Cont::value_type)
    //     };
    // }

    // template <>
    // ir::constant *onnx_importer::emplace_constant<onnx::TensorProto>(const std::optional<onnx::TensorProto> &v);
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
    xt::xarray<float> onnx_importer::convert_to<xt::xarray<float>>(const onnx::TensorProto &tensor);
}