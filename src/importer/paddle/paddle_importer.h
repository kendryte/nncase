/* Copyright 2019 Canaan Inc.
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
#include "framework.pb.h"
#include <filesystem>
#include <ir/connectors.h>
#include <ir/graph.h>
#include <ir/op_utils.h>
#include <ir/ops/transpose.h>
#include <unordered_map>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace importer
{
    class paddle_importer
    {
    public:
        paddle_importer(xtl::span<const uint8_t> model, const std::filesystem::path &params_dir, ir::graph &graph);

        void import();

    private:
        //void convert_op(const tflite::Operator &op);

    private:
        paddle::framework::proto::ProgramDesc model_;
        std::filesystem::path params_dir_;
        ir::graph &graph_;
        std::unordered_map<ir::input_connector *, int32_t> input_tensors_;
        std::unordered_map<int32_t, ir::output_connector *> output_tensors_;
    };
}
}
