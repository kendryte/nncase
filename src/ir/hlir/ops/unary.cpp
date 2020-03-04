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
#include <hlir/ops/unary.h>
#include <llir/ops/unary.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::hlir;

unary::unary(unary_op_t unary_op, shape_t input_shape)
    : unary_op_(unary_op)
{
    add_input("input", dt_float32, input_shape);
    add_output("output", dt_float32, input_shape);
}

void unary::compile(hlir_compile_context &context)
{
    auto l_c = context.graph.emplace<llir::unary>(unary_op(), input().shape());
    context.add_input(input(), l_c->input());
    context.add_output(output(), l_c->output());
}
