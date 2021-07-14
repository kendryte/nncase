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

    [[maybe_unused]] dequantize *deq;

    auto tp1 = graph_.emplace<transpose>(to_data_type(input.type()), get_shape(input.shape()), axis_t { 0, 3, 1, 2 });
    tp1->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/pre_trans");
    auto mid_output = &tp1->output();
    if (input.type() != tflite::TensorType_FLOAT32)
    {
        deq = graph_.emplace<dequantize>(tp1->output().type(), tp1->output().shape(), dt_float32,
            to_quant_param(input.quantization()));
        deq->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/deq");
        mid_output = &deq->output();
        deq->input().connect(tp1->output());
    }

    auto q = graph_.emplace<quantize>(dt_float32, mid_output->shape(), to_data_type(output.type()),
        to_quant_param(output.quantization()));
    q->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/q");
    auto tp2 = graph_.emplace<transpose>(q->output().type(), q->output().shape(), axis_t { 0, 2, 3, 1 });
    tp2->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/trans");

    q->input().connect(*mid_output);
    tp2->input().connect(q->output());
    link_input_tensor(&tp1->input(), op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), &tp2->output());
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

    [[maybe_unused]] dequantize *deq;
    [[maybe_unused]] quantize *q;

    auto tp1 = graph_.emplace<transpose>(to_data_type(input.type()), get_shape(input.shape()), axis_t { 0, 3, 1, 2 });
    tp1->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/pre_trans");
    auto mid_output = &tp1->output();
    //    auto mid_input = &tp1->output();
    if (input.type() != tflite::TensorType_FLOAT32)
    {
        deq = graph_.emplace<dequantize>(tp1->output().type(), tp1->output().shape(), dt_float32,
            to_quant_param(input.quantization()));
        deq->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/deq");
        mid_output = &deq->output();
        deq->input().connect(tp1->output());
    }

    if (output.type() != tflite::TensorType_FLOAT32)
    {
        q = graph_.emplace<quantize>(dt_float32, mid_output->shape(), to_data_type(output.type()),
            to_quant_param(output.quantization()));
        q->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/q");
        mid_output = &q->output();
        q->input().connect(tp1->output());
    }
    auto tp2 = graph_.emplace<transpose>(mid_output->type(), mid_output->shape(), axis_t { 0, 2, 3, 1 });
    tp2->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/trans");

    tp2->input().connect(*mid_output);
    link_input_tensor(&tp1->input(), op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), &tp2->output());
}