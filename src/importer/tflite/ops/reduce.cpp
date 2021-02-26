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
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/transpose.h>
#include <schema_generated.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(REDUCE_MAX)
{
    convert_reduce(op, reduce_max, std::numeric_limits<float>::lowest());
}

DEFINE_TFLITE_LOWER(MEAN)
{
    convert_reduce(op, reduce_mean, 0.f);
}

DEFINE_TFLITE_LOWER(REDUCE_MIN)
{
    convert_reduce(op, reduce_min, std::numeric_limits<float>::max());
}

DEFINE_TFLITE_LOWER(SUM)
{
    convert_reduce(op, reduce_sum, 0.f);
}

void tflite_importer::convert_reduce(const tflite::Operator &op, reduce_op_t reduce_op, float init_value)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto axis = load_axis<int32_t>(get_tensor(op.inputs(), 1));
    auto &output = get_tensor(op.outputs(), 0);
    auto &options = *op.builtin_options_as_ReducerOptions();

    [[maybe_unused]] dequantize *input_dequant;
    [[maybe_unused]] quantize *output_quant;

    auto node = graph_.emplace<nncase::ir::reduce>(reduce_op, get_shape(input.shape()), std::move(axis), init_value, options.keep_dims());
    node->name(get_tensor(op.outputs(), 0).name()->string_view());

    //input dequant
    if (input.type() != tflite::TensorType_FLOAT32)
    {
        quant_param_t input_dequant_paras;
        input_dequant_paras.scale = to_vector(*input.quantization()->scale());
        input_dequant_paras.zero_point = to_vector(*input.quantization()->zero_point());

        input_dequant = graph_.emplace<dequantize>(to_data_type(input.type()), get_shape(input.shape()), dt_float32, input_dequant_paras);
        input_dequant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/input_dequant");
        node->input().connect(input_dequant->output());
        input_tensors_.emplace(&input_dequant->input(), op.inputs()->Get(0));
    }
    else
    {
        input_tensors_.emplace(&node->input(), op.inputs()->Get(0));
    }

    //output dequant
    if (node->output().type() != to_data_type(input.type()))
    {
        quant_param_t output_quant_paras;
        output_quant_paras.scale = to_vector(*output.quantization()->scale());
        output_quant_paras.zero_point = to_vector(*output.quantization()->zero_point());

        output_quant = graph_.emplace<quantize>(dt_float32, get_shape(output.shape()), to_data_type(output.type()), output_quant_paras);
        output_quant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/output_quant");
        output_quant->input().connect(node->output());
        output_tensors_.emplace(op.outputs()->Get(0), &output_quant->output());
    }
    else
    {
        output_tensors_.emplace(op.outputs()->Get(0), &node->output());
    }
}
