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
#include <filesystem>
#include <nncase/ir/graph.h>
#include <span>
#include <unordered_map>
#include <vector>

namespace nncase::importer
{
struct import_options
{
    std::string input_layout = "NCHW";
    std::string output_layout = "NCHW";
    std::span<const std::string> output_arrays;
};

void import_tflite(ir::graph &graph, std::span<const uint8_t> model, const import_options &options);
void import_onnx(ir::graph &graph, std::span<const uint8_t> model, const import_options &options);
}
