/* Copyright 2019-2021 Canaan Inc.
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

#include "ir.h"
#include <cstdint>
#include <nncase/importer/importer.h>
#include <nncase/importer/util.h>
#include <nncase/ir/connectors.h>
#include <nncase/ir/debug.h>
#include <nncase/ir/graph.h>
#include <nncase/ir/ir_types.h>
#include <nncase/ir/node.h>
#include <nncase/ir/op_utils.h>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <xtensor/xadapt.hpp>

namespace nncase
{
namespace importer
{
    class pnnx_importer
    {
    public:
        pnnx_importer(std::string parampath, std::string binpath, ir::graph &graph);

        void import(const struct import_options &options, std::string &real_inlayout, std::string &real_outlayout);

    private:
        void convert_op(const pnnx::Operator &op);

#define DEFINE_OPCODE(opcode, opcode2) void convert_op_##opcode2(const pnnx::Operator &op);
#include "opcode.def"
#undef DEFINE_OPCODE

    private:
        ir::graph &graph_;
        pnnx::Graph pnnx_graph_;
        std::unordered_map<ir::input_connector *, std::string> input_tensors_;
        std::unordered_map<std::string, ir::output_connector *> output_tensors_;
    };
}
}

#define DEFINE_PNNX_LOWER(opcode) \
    void nncase::importer::pnnx_importer::convert_op_##opcode(const pnnx::Operator &op)
