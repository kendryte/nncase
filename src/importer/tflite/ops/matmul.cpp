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
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/matmul.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(FULLY_CONNECTED)
{
    auto &input_a = get_tensor(op.inputs(), 0);
    auto &input_b = get_tensor(op.inputs(), 1);
    [[maybe_unused]] int not_f32 = 0;

    auto &output = get_tensor(op.outputs(), 0);
    auto &options = *op.builtin_options_as_FullyConnectedOptions();

    assert(options.weights_format() == tflite::FullyConnectedOptionsWeightsFormat_DEFAULT);

    dequantize *input_a_dequant, *input_b_dequant, *bias_dequant;
    transpose *input_b_trans;
    quantize *output_quant;

    // input_a dequantize
    if (input_a.type() != tflite::TensorType_FLOAT32)
    {
        not_f32 = 1;
        quant_param_t input_a_dequant_paras;
        input_a_dequant_paras.scale = to_vector(*input_a.quantization()->scale());
        input_a_dequant_paras.zero_point = to_vector(*input_a.quantization()->zero_point());

        input_a_dequant = graph_.emplace<dequantize>(to_data_type(input_a.type()), get_shape(input_a.shape()), dt_float32, 
            input_a_dequant_paras);
        input_a_dequant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/input_a_dequant");
    }

    // input_b dequantize
    if (input_b.type() != tflite::TensorType_FLOAT32)
    {
        quant_param_t input_b_dequant_paras;
        input_b_dequant_paras.scale = to_vector(*input_b.quantization()->scale());
        input_b_dequant_paras.zero_point = to_vector(*input_b.quantization()->zero_point());

        input_b_dequant = graph_.emplace<dequantize>(to_data_type(input_b.type()), get_shape(input_b.shape()), dt_float32, input_b_dequant_paras);
        input_b_dequant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/input_b_dequant");
        input_b_trans = graph_.emplace<transpose>(input_b_dequant->output().type(), get_shape(input_b.shape()), axis_t { 1, 0 });
        input_b_trans->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/input_b_transpose_0");
        input_b_trans->input().connect(input_b_dequant->output());
        link_input_tensor(&input_b_dequant->input(), op.inputs()->Get(1));
    }
    else
    {
        input_b_trans = graph_.emplace<transpose>(to_data_type(input_b.type()), get_shape(input_b.shape()), axis_t { 1, 0 });
        input_b_trans->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/input_b_transpose_1");
        link_input_tensor(&input_b_trans->input(), op.inputs()->Get(1));
    }

    auto rshape = graph_.emplace<bitcast>(dt_float32, get_shape(input_a.shape()), dt_float32,
        axis_t { -1, (int32_t)input_b_trans->output().shape()[0] });
    auto fc = graph_.emplace<matmul>(rshape->output().shape(), input_b_trans->output().shape(),
        to_float_clamp_range(options.fused_activation_function()));

    // bias dequantize
    if (op.inputs()->size() == 3)
    {
        auto &bias = get_tensor(op.inputs(), 2);
        if (bias.type() != tflite::TensorType_FLOAT32)
        {
            quant_param_t bias_dequant_paras;
            bias_dequant_paras.scale = to_vector(*bias.quantization()->scale());
            bias_dequant_paras.zero_point = to_vector(*bias.quantization()->zero_point());

            bias_dequant = graph_.emplace<dequantize>(to_data_type(bias.type()), get_shape(bias.shape()), dt_float32, bias_dequant_paras);
            bias_dequant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/bias_quant");
            fc->bias().connect(bias_dequant->output());
            link_input_tensor(&bias_dequant->input(), op.inputs()->Get(2));
        }
        else
        {
            link_input_tensor(&fc->bias(), op.inputs()->Get(2));
        }
    }

    // input_a?dequant connect
    if (not_f32)
    {
        rshape->input().connect(input_a_dequant->output());
        link_input_tensor(&input_a_dequant->input(), op.inputs()->Get(0));
    }
    else
    {
        link_input_tensor(&rshape->input(), op.inputs()->Get(0));
    }

    rshape->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/reshape");
    fc->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/fc");

    fc->input_a().connect(rshape->output());
    fc->input_b().connect(input_b_trans->output());

    if (fc->output().type() != to_data_type(output.type()))
    {
        quant_param_t output_quant_paras;
        output_quant_paras.scale = to_vector(*output.quantization()->scale());
        output_quant_paras.zero_point = to_vector(*output.quantization()->zero_point());

        output_quant = graph_.emplace<quantize>(dt_float32, fc->output().shape(), to_data_type(output.type()), output_quant_paras);
        output_quant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/output_quant");
        output_quant->input().connect(fc->output());
        link_output_tensor(op.outputs()->Get(0), &output_quant->output());
    }
    else
    {
        link_output_tensor(op.outputs()->Get(0), &fc->output());
    }
}
