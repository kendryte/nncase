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
#include <hlir/op_utils.h>
#include <hlir/ops/fused_unary.h>
#include <llir/ops/memory_copy.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::hlir;

fused_unary::fused_unary(graph subgraph)
    : subgraph_(std::move(subgraph))
{
    if (subgraph_.inputs().size() != 1
        || subgraph_.outputs().size() != 1)
        throw std::invalid_argument("Invalid subgraph inouts for fused unary");

    if (subgraph_.inputs()[0]->output().shape() != subgraph_.outputs()[0]->input().shape())
        throw std::invalid_argument("Invalid subgraph shape for fused unary");

    add_input("input", dt_float32, subgraph_.inputs()[0]->output().shape());
    add_output("output", dt_float32, subgraph_.outputs()[0]->input().shape());
}

void fused_unary::compile(hlir_compile_context &context)
{
    throw std::runtime_error("Not implemented");
}
