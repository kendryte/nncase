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
#include "paddle_importer.h"
#include <importer/importer.h>
#include <ir/ops/constant.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

paddle_importer::paddle_importer(xtl::span<const uint8_t> model, const std::filesystem::path &params_dir, ir::graph &graph)
    : graph_(graph)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if (!model_.ParseFromArray(model.data(), (int)model.size()))
        throw std::runtime_error("Invalid PaddlePaddle model");
}

void paddle_importer::import()
{
}

graph nncase::importer::import_paddle(xtl::span<const uint8_t> model, const std::filesystem::path &params_dir)
{
    graph graph;
    paddle_importer(model, params_dir, graph).import();
    return graph;
}
