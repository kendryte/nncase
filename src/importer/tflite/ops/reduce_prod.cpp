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
#include <nncase/ir/ops/reduce_prod.h>
#include <nncase/ir/ops/transpose.h>
#include <schema_generated.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(REDUCE_PROD)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto axis = load_axis<int32_t>(get_tensor(op.inputs(), 1));
    auto &output = get_tensor(op.outputs(), 0);
    auto &options = *op.builtin_options_as_ReducerOptions();

    auto input_type = to_data_type(input.type());
    auto node = graph_.emplace<reduce_prod>(input_type, get_shape(input.shape()), std::move(axis), options.keep_dims());
    node->name(output.name()->string_view());

    //input dequant
    if (input_type == dt_uint8 || input_type == dt_int8)
    {
        quant_param_t input_dequant_paras = to_quant_param(input.quantization());
        auto input_dequant = graph_.emplace<dequantize>(input_type, get_shape(input.shape()), dt_float32, input_dequant_paras);
        input_dequant->name(std::string(output.name()->string_view()) + "/input_dequant");
        node->input().connect(input_dequant->output());
        input_tensors_.emplace(&input_dequant->input(), op.inputs()->Get(0));
    }
    else
    {
        input_tensors_.emplace(&node->input(), op.inputs()->Get(0));
    }

    //output dequant
    if (node->output().type() != input_type)
    {
        quant_param_t output_quant_paras = to_quant_param(output.quantization());
        auto output_quant = graph_.emplace<quantize>(dt_float32, get_shape(output.shape()), to_data_type(output.type()), output_quant_paras);
        output_quant->name(std::string(output.name()->string_view()) + "/output_quant");
        output_quant->input().connect(node->output());
        output_tensors_.emplace(op.outputs()->Get(0), &output_quant->output());
    }
    else
    {
        output_tensors_.emplace(op.outputs()->Get(0), &node->output());
    }
}
