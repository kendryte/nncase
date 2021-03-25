/* Copyright 2020 Canaan Inc.
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
#include "../tflite_importer.h"
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(SOFTMAX)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &output = get_tensor(op.outputs(), 0);
    auto &options = *op.builtin_options_as_SoftmaxOptions();

    dequantize *input_dequant;
    quantize *output_quant;

    auto in_shape = get_shape(input.shape());
    axis_t reduce_axis;
    reduce_axis.push_back(int32_t(in_shape.size() - 1));

    auto max = graph_.emplace<reduce>(reduce_max, in_shape, reduce_axis, std::numeric_limits<float>::lowest(), true);
    auto sub = graph_.emplace<binary>(binary_sub, in_shape, max->output().shape(), value_range<float>::full());
    auto beta = graph_.emplace<constant>(float(options.beta()));
    auto mul = graph_.emplace<binary>(binary_mul, sub->output().shape(), beta->output().shape(), value_range<float>::full());
    auto exp = graph_.emplace<unary>(unary_exp, mul->output().shape());
    auto sum = graph_.emplace<reduce>(reduce_sum, exp->output().shape(), reduce_axis, 0.f, true);
    auto div = graph_.emplace<binary>(binary_div, exp->output().shape(), sum->output().shape(), value_range<float>::full());

    max->name(get_tensor(op.outputs(), 0).name()->string_view());
    sub->name(get_tensor(op.outputs(), 0).name()->string_view());
    beta->name(get_tensor(op.outputs(), 0).name()->string_view());
    mul->name(get_tensor(op.outputs(), 0).name()->string_view());
    exp->name(get_tensor(op.outputs(), 0).name()->string_view());
    sum->name(get_tensor(op.outputs(), 0).name()->string_view());
    div->name(get_tensor(op.outputs(), 0).name()->string_view());

    if (input.type() != tflite::TensorType_FLOAT32)
    {
        quant_param_t input_dequant_paras = to_quant_param(input.quantization());
        input_dequant = graph_.emplace<dequantize>(to_data_type(input.type()), get_shape(input.shape()), dt_float32, input_dequant_paras);
        input_dequant->name(get_tensor(op.outputs(), 0).name()->string_view());
        max->input().connect(input_dequant->output());
        sub->input_a().connect(input_dequant->output());
        link_input_tensor(&input_dequant->input(), op.inputs()->Get(0));
        //        link_input_tensor(&sub->input_a(), op.inputs()->Get(0));

    }
    else
    {
        link_input_tensor(&max->input(), op.inputs()->Get(0));
        link_input_tensor(&sub->input_a(), op.inputs()->Get(0));
    }

    sub->input_b().connect(max->output());
    mul->input_a().connect(sub->output());
    mul->input_b().connect(beta->output());
    exp->input().connect(mul->output());
    sum->input().connect(exp->output());
    div->input_a().connect(exp->output());
    div->input_b().connect(sum->output());

    if (div->output().type() != to_data_type(input.type()))
    {
        quant_param_t output_quant_paras = to_quant_param(output.quantization());
        output_quant = graph_.emplace<quantize>(dt_float32, get_shape(output.shape()), to_data_type(output.type()), output_quant_paras);
        output_quant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/div_output_quant");
        output_quant->input().connect(div->output());
        link_output_tensor(op.outputs()->Get(0), &output_quant->output());
    }
    else
    {
        link_output_tensor(op.outputs()->Get(0), &div->output());
    }
}
