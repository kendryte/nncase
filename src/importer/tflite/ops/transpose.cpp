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
#include <nncase/ir/ops/transpose.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(TRANSPOSE)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &output = get_tensor(op.outputs(), 0);
    [[maybe_unused]] auto &options = *op.builtin_options_as_TransposeOptions();
    auto perm = load_axis<int32_t>(get_tensor(op.inputs(), 1));

    [[maybe_unused]] dequantize *deq = nullptr;
    [[maybe_unused]] quantize *q = nullptr;
    auto in_type = to_data_type(input.type());
    if (in_type != dt_float32)
    {
        quant_param_t in_deq_params = to_quant_param(output.quantization());
        deq = graph_.emplace<dequantize>(in_type, get_shape(input.shape()), dt_float32, in_deq_params);
        in_type = deq->output().type();

        quant_param_t out_q_params = to_quant_param(output.quantization());
        q = graph_.emplace<quantize>(dt_float32, get_shape(output.shape()), to_data_type(output.type()), out_q_params);
    }

    auto node = graph_.emplace<transpose>(in_type, get_shape(input.shape()), perm);
    node->name(get_tensor(op.outputs(), 0).name()->string_view());

    if (deq)
    {
        deq->name(node->name() + "/in_dequant");
        q->name(node->name() + "/out_quant");

        node->input().connect(deq->output());
        q->input().connect(node->output());

        link_input_tensor(&deq->input(), op.inputs()->Get(0));
        link_output_tensor(op.outputs()->Get(0), &q->output());
    }
    else
    {
        link_input_tensor(&node->input(), op.inputs()->Get(0));
        link_output_tensor(op.outputs()->Get(0), &node->output());
    }
}
