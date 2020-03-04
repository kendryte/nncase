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
#include <hlir/op_utils.h>
#include <hlir/ops/table_lookup.h>
#include <llir/ops/table_lookup.h>

using namespace nncase;
using namespace nncase::hlir;

table_lookup1d::table_lookup1d(datatype_t type, shape_t input_shape, size_t table_size)
{
    add_input("input", type, input_shape);
    add_input("table", type, shape_t { table_size });
    add_output("output", type, input_shape);
}

void table_lookup1d::compile(hlir_compile_context &context)
{
    auto l_lut = context.graph.emplace<llir::table_lookup1d>(input().type(), input().shape(), table().shape()[0]);

    context.add_input(input(), l_lut->input());
    context.add_input(table(), l_lut->table());
    context.add_output(output(), l_lut->output());
}
