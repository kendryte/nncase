/* Copyright 2019-2021 Canaan Inc.
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
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(ADD)
{
    convert_binary(op, binary_add, op.builtin_options_as_AddOptions()->fused_activation_function());
}

DEFINE_TFLITE_LOWER(DIV)
{
    convert_binary(op, binary_div, op.builtin_options_as_DivOptions()->fused_activation_function());
}

DEFINE_TFLITE_LOWER(FLOOR_DIV)
{
    auto &input_a = get_tensor(op.inputs(), 0);
    auto &input_b = get_tensor(op.inputs(), 1);
    auto &output = get_tensor(op.outputs(), 0);

    auto div = graph_.emplace<binary>(binary_div, to_data_type(input_a.type()), get_shape(input_a.shape()), get_shape(input_b.shape()), value_range<float>::full());
    div->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/binary");

    // input_a dequantize
    if (input_a.type() != tflite::TensorType_FLOAT32)
    {
        quant_param_t input_a_paras = to_quant_param(input_a.quantization());
        auto input_a_dequant = graph_.emplace<dequantize>(to_data_type(input_a.type()), get_shape(input_a.shape()), dt_float32, input_a_paras);
        input_a_dequant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/input_a_dequant");
        div->input_a().connect(input_a_dequant->output());
        link_input_tensor(&input_a_dequant->input(), op.inputs()->Get(0));
    }
    else
    {
        link_input_tensor(&div->input_a(), op.inputs()->Get(0));
    }

    // input_b dequantize
    if (input_b.type() != tflite::TensorType_FLOAT32)
    {
        quant_param_t input_b_paras = to_quant_param(input_b.quantization());
        auto input_b_dequant = graph_.emplace<dequantize>(to_data_type(input_b.type()), get_shape(input_b.shape()), dt_float32, input_b_paras);
        input_b_dequant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/input_b_dequant");
        div->input_b().connect(input_b_dequant->output());
        link_input_tensor(&input_b_dequant->input(), op.inputs()->Get(1));
    }
    else
    {
        link_input_tensor(&div->input_b(), op.inputs()->Get(1));
    }

    auto floor = graph_.emplace<unary>(unary_floor, div->output().shape());
    floor->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/unary");
    floor->input().connect(div->output());

    // output quantize
    if (floor->output().type() != to_data_type(output.type()))
    {
        quant_param_t output_paras = to_quant_param(output.quantization());
        auto output_quant = graph_.emplace<quantize>(dt_float32, get_shape(output.shape()), to_data_type(output.type()), output_paras);
        output_quant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/output_quant");
        output_quant->input().connect(floor->output());
        link_output_tensor(op.outputs()->Get(0), &output_quant->output());
    }
    else
    {
        link_output_tensor(op.outputs()->Get(0), &floor->output());
    }
}

DEFINE_TFLITE_LOWER(FLOOR_MOD)
{
    convert_binary(op, binary_floor_mod, tflite::ActivationFunctionType_NONE);
}

DEFINE_TFLITE_LOWER(MAXIMUM)
{
    convert_binary(op, binary_max, tflite::ActivationFunctionType_NONE);
}

DEFINE_TFLITE_LOWER(MINIMUM)
{
    convert_binary(op, binary_min, tflite::ActivationFunctionType_NONE);
}

DEFINE_TFLITE_LOWER(MUL)
{
    convert_binary(op, binary_mul, op.builtin_options_as_MulOptions()->fused_activation_function());
}

DEFINE_TFLITE_LOWER(POW)
{
    convert_binary(op, binary_pow, tflite::ActivationFunctionType_NONE);
}

DEFINE_TFLITE_LOWER(SUB)
{
    convert_binary(op, binary_sub, op.builtin_options_as_SubOptions()->fused_activation_function());
}

void tflite_importer::convert_binary(const tflite::Operator &op, binary_op_t binary_op, tflite::ActivationFunctionType activation)
{
    auto &input_a = get_tensor(op.inputs(), 0);
    auto input_type = to_data_type(input_a.type());
    auto &input_b = get_tensor(op.inputs(), 1);
    auto &output = get_tensor(op.outputs(), 0);

    auto add = graph_.emplace<binary>(binary_op, input_type, get_shape(input_a.shape()), get_shape(input_b.shape()), to_float_clamp_range(activation));
    add->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/binary");

    dequantize *input_a_dequant, *input_b_dequant;
    quantize *output_quant;
    // input_a dequantize
    if (input_type == dt_uint8 || input_type == dt_int8)
    {
        quant_param_t input_a_paras = to_quant_param(input_a.quantization());
        input_a_dequant = graph_.emplace<dequantize>(to_data_type(input_a.type()), get_shape(input_a.shape()), dt_float32, input_a_paras);
        input_a_dequant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/input_a_dequant");
        add->input_a().connect(input_a_dequant->output());
        link_input_tensor(&input_a_dequant->input(), op.inputs()->Get(0));
    }
    else
    {
        link_input_tensor(&add->input_a(), op.inputs()->Get(0));
    }

    //input_b dequantize
    if (input_type == dt_uint8 || input_type == dt_int8)
    {
        quant_param_t input_b_paras = to_quant_param(input_b.quantization());
        input_b_dequant = graph_.emplace<dequantize>(to_data_type(input_b.type()), get_shape(input_b.shape()), dt_float32, input_b_paras);
        input_b_dequant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/input_b_dequant");
        add->input_b().connect(input_b_dequant->output());
        link_input_tensor(&input_b_dequant->input(), op.inputs()->Get(1));
    }
    else
    {
        link_input_tensor(&add->input_b(), op.inputs()->Get(1));
    }

    //output quantize
    if (add->output().type() != to_data_type(output.type()))
    {
        quant_param_t output_paras = to_quant_param(output.quantization());
        output_quant = graph_.emplace<quantize>(dt_float32, get_shape(output.shape()), to_data_type(output.type()), output_paras);
        output_quant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/output_quant");
        output_quant->input().connect(add->output());
        link_output_tensor(op.outputs()->Get(0), &output_quant->output());
    }
    else
    {
        link_output_tensor(op.outputs()->Get(0), &add->output());
    }
}
