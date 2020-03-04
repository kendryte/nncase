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
#include <hlir/ops/clamp.h>
#include <llir/ops/binary.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::hlir;

clamp::clamp(shape_t input_shape, shape_t input_low_shape, shape_t input_high_shape)
{
    add_input("input", dt_float32, input_shape);
    add_input("input_low", dt_float32, input_low_shape);
    add_input("input_high", dt_float32, input_high_shape);
    add_output("output", dt_float32, input_shape);
}

void clamp::compile(hlir_compile_context &context)
{
    auto l_max = context.graph.emplace<llir::binary>(binary_max, input().shape(), input_low().shape(), value_range<float>::full());
    auto l_min = context.graph.emplace<llir::binary>(binary_min, l_max->output().shape(), input_high().shape(), value_range<float>::full());
    l_min->input_a().connect(l_max->output());

    context.add_input(input(), l_max->input_a());
    context.add_input(input_low(), l_max->input_b());
    context.add_input(input_high(), l_min->input_b());
    context.add_output(output(), l_min->output());
}
