/* Copyright 2019-2020 Canaan Inc.
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
#include <hlir/placeholders.h>
#include <llir/placeholders.h>

using namespace nncase;
using namespace nncase::hlir;

void input_node::compile(hlir_compile_context &context)
{
    auto l_input = context.graph.emplace<llir::input_node>(output().type(), output().shape(), output().memory_type());
    l_input->name(name());
    context.add_output(output(), l_input->output());
}

void output_node::compile(hlir_compile_context &context)
{
    auto l_output = context.graph.emplace<llir::output_node>(input().type(), input().shape());
    l_output->name(name());
    context.add_input(input(), l_output->input());
}

void ignore_node::compile(hlir_compile_context &context)
{
    auto l_ignore = context.graph.emplace<llir::ignore_node>(input().type(), input().shape());
    context.add_input(input(), l_ignore->input());
}
