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
#include <hlir/ops/dequantize.h>
#include <llir/ops/dequantize.h>
#include <hlir/op_utils.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::hlir;

dequantize::dequantize(shape_t input_shape, quant_param_t quant_param)
    : quant_param_(quant_param)
{
    add_input("input", dt_uint8, input_shape);
    add_output("output", dt_float32, input_shape);
}

void dequantize::compile(hlir_compile_context &context)
{
    auto l_c = context.graph.emplace<llir::dequantize>(input().shape(), quant_param());
    context.add_input(input(), l_c->input());
    context.add_output(output(), l_c->output());
}
