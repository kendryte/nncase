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
        void convert_op(const onnx::NodeProto &node);
#define DEFINE_OPCODE(opcode) void convert_op_##opcode(const onnx::NodeProto &node);
#include "opcode.def"
#undef DEFINE_OPCODE

        nncase::hlir::shape_t get_shape(const std::string &value) const;
        static nncase::hlir::shape_t get_shape(const onnx::ValueInfoProto &value);
        static nncase::datatype_t get_datatype(const onnx::ValueInfoProto &value);
        static nncase::datatype_t get_datatype(const onnx::TensorProto_DataType datatype);
        static std::string to_string(const onnx::TensorProto_DataType datatype);

        hlir::graph &graph_;
        onnx::ModelProto model_;

        std::unordered_map<hlir::input_connector *, std::string_view> input_tensors_;
        std::unordered_map<std::string_view, hlir::output_connector *> output_tensors_;
    };
}
}
