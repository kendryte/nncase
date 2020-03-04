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
#include <hlir/ops/reduce_window2d.h>
#include <llir/ops/reduce_window2d.h>
#include <hlir/op_utils.h>

using namespace nncase;
using namespace nncase::hlir;

reduce_window2d::reduce_window2d(reduce_op_t reduce_op, shape_t input_shape, float init_value, int32_t filter_h, int32_t filter_w, padding padding_h, padding padding_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation)
    : reduce_op_(reduce_op), init_value_(init_value), filter_h_(filter_h), filter_w_(filter_w), padding_h_(padding_h), padding_w_(padding_w), stride_h_(stride_h), stride_w_(stride_w), dilation_h_(dilation_h), dilation_w_(dilation_w), fused_activation_(fused_activation)
{
    add_input("input", dt_float32, input_shape);
    add_output("output", dt_float32,
        shape_t {
            input_shape[0],
            input_shape[1],
            get_windowed_output_size((int32_t)input_shape[2] + padding_h_.sum(), filter_h_, stride_h_, dilation_h_, false),
            get_windowed_output_size((int32_t)input_shape[3] + padding_w_.sum(), filter_w_, stride_w_, dilation_w_, false) });
}

void reduce_window2d::compile(hlir_compile_context &context)
{
    auto l_c = context.graph.emplace<llir::reduce_window2d>(reduce_op(), input().shape(), init_value(), filter_h(), filter_w(), padding_h(), padding_w(),
		stride_h(), stride_w(), dilation_h(), dilation_w(), fused_activation());
    context.add_input(input(), l_c->input());
    context.add_output(output(), l_c->output());
}
