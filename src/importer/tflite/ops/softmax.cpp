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
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/softmax.h>
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
    auto input_type = to_data_type(input.type());

    auto sm = graph_.emplace<softmax>(input_type, in_shape, static_cast<int32_t>(in_shape.size() - 1), options.beta());
    sm->name(output.name()->string_view());

    if (input_type != dt_float32)
    {
        quant_param_t input_dequant_paras = to_quant_param(input.quantization());
        input_dequant = graph_.emplace<dequantize>(to_data_type(input.type()), get_shape(input.shape()), dt_float32, input_dequant_paras);
        input_dequant->name(get_tensor(op.outputs(), 0).name()->string_view());
        sm->input().connect(input_dequant->output());
        link_input_tensor(&input_dequant->input(), op.inputs()->Get(0));
    }
    else
    {
        link_input_tensor(&sm->input(), op.inputs()->Get(0));
    }

    if (sm->output().type() != to_data_type(input.type()))
    {
        quant_param_t output_quant_paras = to_quant_param(output.quantization());
        output_quant = graph_.emplace<quantize>(dt_float32, get_shape(output.shape()), to_data_type(output.type()), output_quant_paras);
        output_quant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/softmax_output_quant");
        output_quant->input().connect(sm->output());
        link_output_tensor(op.outputs()->Get(0), &output_quant->output());
    }
    else
    {
        link_output_tensor(op.outputs()->Get(0), &sm->output());
    }
}
