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
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/dequantize.h>
#include <nncase/ir/ops/quantize.h>
#include <nncase/ir/ops/transpose.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(QUANTIZE)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &output = get_tensor(op.outputs(), 0);

    auto q = graph_.emplace<quantize>(to_data_type(input.type()), get_shape(input.shape()), to_data_type(output.type()),
        to_quant_param(output.quantization()));
    q->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/q");

    link_input_tensor(&q->input(), op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), &q->output());
}

DEFINE_TFLITE_LOWER(FAKE_QUANT)
{
    auto &input = get_tensor(op.inputs(), 0);

    auto in_shape = get_shape(input.shape());

    auto nop = graph_.emplace<bitcast>(to_data_type(input.type()), in_shape, in_shape);
    nop->name(get_tensor(op.outputs(), 0).name()->string_view());

    link_input_tensor(&nop->input(), op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), &nop->output());
}

DEFINE_TFLITE_LOWER(DEQUANTIZE)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &output = get_tensor(op.outputs(), 0);

    if (op.outputs()->size() != 0)
    {
        auto deq = graph_.emplace<dequantize>(to_data_type(input.type()), get_shape(input.shape()), to_data_type(output.type()),
            to_quant_param(input.quantization()));
        deq->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/deq");
        link_input_tensor(&deq->input(), op.inputs()->Get(0));
        link_output_tensor(op.outputs()->Get(0), &deq->output());
    }
}