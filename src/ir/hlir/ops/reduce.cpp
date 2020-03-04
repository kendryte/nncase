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
#include <hlir/ops/reduce.h>
#include <llir/ops/reduce.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::hlir;

reduce::reduce(reduce_op_t reduce_op, shape_t input_shape, axis_t axis, float init_value, bool keep_dims)
    : reduce_op_(reduce_op), keep_dims_(keep_dims), axis_(normalize_reduce_axis(input_shape, axis)), init_value_(init_value)
{
    add_input("input", dt_float32, input_shape);
    add_output("output", dt_float32, get_reduced_shape(input_shape, axis_, keep_dims_));
}

void reduce::compile(hlir_compile_context &context)
{
    auto l_c = context.graph.emplace<llir::reduce>(reduce_op(), input().shape(), axis(), init_value(), keep_dims());
    context.add_input(input(), l_c->input());
    context.add_output(output(), l_c->output());
}
