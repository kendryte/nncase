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

#include "layer.h"
#include "modelbin.h"
#include "paramdict.h"
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
    class ncnn_importer
    {
    public:
        ncnn_importer(const std::filesystem::path &paramfilename, const std::filesystem::path &binfilename, ir::graph &graph);

        void import(const import_options &options);

    private:
        void convert_op(const ncnn::Layer &layer, const ncnn::ParamDict &pd, const ncnn::ModelBin &mb);

#define DEFINE_OPCODE(opcode) void convert_op_##opcode(const ncnn::Layer &layer, const ncnn::ParamDict &pd, const ncnn::ModelBin &mb);
#include "opcode.def"
#undef DEFINE_OPCODE

    private:
        ir::graph &graph_;
        std::filesystem::path paramfilename;
        std::filesystem::path binfilename;
        std::unordered_map<ir::input_connector *, std::string> input_tensors_;
        std::unordered_map<std::string, ir::output_connector *> output_tensors_;
    };
}
}

#define DEFINE_NCNN_LOWER(opcode) \
    void nncase::importer::ncnn_importer::convert_op_##opcode(const ncnn::Layer &layer, const ncnn::ParamDict &pd, const ncnn::ModelBin &mb)
