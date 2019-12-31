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
    std::unordered_map<hlir::output_connector *, llir::output_connector *> inputs;
    std::unordered_map<hlir::input_connector *, llir::input_connector *> outputs;

    auto l_i = context.graph.emplace<llir::memory_copy>(input().type(), input().shape(), input().type(), input().shape());
    context.add_input(input(), l_i->input());
    inputs.emplace(&subgraph_.inputs()[0]->output(), &l_i->output());

    auto l_o = context.graph.emplace<llir::memory_copy>(output().type(), output().shape(), output().type(), output().shape());
    context.add_output(output(), l_o->output());
    outputs.emplace(&subgraph_.outputs()[0]->input(), &l_o->input());
    subgraph_.flatten_subgraph(context, inputs, outputs);
}
