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

#include <unordered_map>
#include <string_view>
#include <variant>
#include <cstdint>

#include <xtensor/xadapt.hpp>
#include "onnx/onnx.pb.h"

#include <hlir/ir_types.h>

namespace nncase
{

namespace hlir
{
    class graph;

    class input_connector;
    class output_connector;
}

namespace importer
{
    class onnx_importer
    {
    public:
        onnx_importer(xtl::span<const std::uint8_t> model, hlir::graph &graph);

        void import();

    private:
        typedef std::variant
        <
            float,
            std::int64_t,
            std::string
        > attribute_value_type;

        void convert_op(const onnx::NodeProto &node);
#define DEFINE_OPCODE(opcode) void convert_op_##opcode(const onnx::NodeProto &node);
#include "opcode.def"
#undef DEFINE_OPCODE

        void convert_unary(const onnx::NodeProto &node, const unary_op_t unary_op);
        void convert_binary(const onnx::NodeProto &node, const binary_op_t binary_op);

        void convert_pool(const onnx::NodeProto &node, const reduce_op_t reduce_op, const float initial_value);

        const onnx::ValueInfoProto* find_value_info(const std::string &value) const;
        nncase::hlir::shape_t get_shape(const std::string &value) const;
        static nncase::hlir::shape_t get_shape(const onnx::ValueInfoProto &value);
        static nncase::datatype_t get_datatype(const onnx::ValueInfoProto &value);
        static nncase::datatype_t get_datatype(const onnx::TensorProto_DataType datatype);
        static nncase::datatype_t get_datatype(const onnx::AttributeProto_AttributeType type);
        template<typename T> static constexpr nncase::datatype_t get_datatype();

        static std::string to_string(const onnx::TensorProto_DataType datatype);
        static std::string to_string(const onnx::AttributeProto_AttributeType type);
        attribute_value_type get_attribute(const onnx::NodeProto &node, const std::string &name) const;
        template<typename T> static std::optional<T> get_attribute(const onnx::NodeProto &node, const std::string &name);
        const onnx::TensorProto& get_initializer(const std::string &name) const;
        template<typename T> static T to(const onnx::TensorProto &tensor);

        hlir::graph &graph_;
        onnx::ModelProto model_;

        std::unordered_map<hlir::input_connector *, std::string_view> input_tensors_;
        std::unordered_map<std::string_view, hlir::output_connector *> output_tensors_;
    };

    template<> constexpr nncase::datatype_t onnx_importer::get_datatype<float>()
    {
        return nncase::dt_float32;
    }

    template<> constexpr nncase::datatype_t onnx_importer::get_datatype<std::int64_t>()
    {
        return nncase::dt_uint8;
    }

    template<> std::optional<float> onnx_importer::get_attribute<float>(const onnx::NodeProto &node, const std::string &name);
    template<> std::optional<std::int64_t> onnx_importer::get_attribute<std::int64_t>(const onnx::NodeProto &node, const std::string &name);
    template<> std::optional<int> onnx_importer::get_attribute<int>(const onnx::NodeProto &node, const std::string &name);
    template<> std::optional<std::string> onnx_importer::get_attribute<std::string>(const onnx::NodeProto &node, const std::string &name);
    template<> std::optional<xtl::span<const float>> onnx_importer::get_attribute<xtl::span<const float>>(const onnx::NodeProto &node, const std::string &name);
    template<> std::optional<xtl::span<const std::int64_t>> onnx_importer::get_attribute<xtl::span<const std::int64_t>>(const onnx::NodeProto &node, const std::string &name);
    template<> std::optional<xtl::span<const std::string>> onnx_importer::get_attribute<xtl::span<const std::string>>(const onnx::NodeProto &node, const std::string &name);

    template<> float onnx_importer::to<float>(const onnx::TensorProto &tensor);
    template<> std::uint8_t onnx_importer::to<std::uint8_t>(const onnx::TensorProto &tensor);
    template<> hlir::axis_t onnx_importer::to<hlir::axis_t>(const onnx::TensorProto &tensor);
    template<> xt::xarray<float> onnx_importer::to<xt::xarray<float>>(const onnx::TensorProto &tensor);
    template<> xt::xarray<std::uint8_t> onnx_importer::to<xt::xarray<std::uint8_t>>(const onnx::TensorProto &tensor);
}
}
