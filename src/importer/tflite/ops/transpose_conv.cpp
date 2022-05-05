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
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/conv2d_transpose.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(TRANSPOSE_CONV)
{
    auto out_shape = nhwc_to_nchw(load_shape<int32_t>(get_tensor(op.inputs(), 0)));
    auto &input = get_tensor(op.inputs(), 2);
    auto &weights = get_tensor(op.inputs(), 1);
    auto &output = get_tensor(op.outputs(), 0);
    auto &options = *op.builtin_options_as_TransposeConvOptions();

    std::vector<float> bias_tensor(out_shape[1], 0.f);

    auto pre_trans = nhwc_to_nchw(dt_float32, get_shape(input.shape()));
    pre_trans->name(get_tensor(op.outputs(), 0).name()->string_view());

    auto weights_trans = graph_.emplace<transpose>(dt_float32, get_shape(weights.shape()), axis_t { 0, 3, 1, 2 });
    weights_trans->name(get_tensor(op.outputs(), 0).name()->string_view());

    shape_t bias_shape = { out_shape[1] };
    auto bias = graph_.emplace<constant>(dt_float32, bias_shape, std::span<const float>(bias_tensor));

    auto out_h = (int32_t)out_shape[2];
    auto out_w = (int32_t)out_shape[3];
    auto f_h = (int32_t)weights_trans->output().shape()[2];
    auto f_w = (int32_t)weights_trans->output().shape()[3];
    auto stride_h = options.stride_h();
    auto stride_w = options.stride_w();
    auto dilation_h = 1;
    auto dilation_w = 1;
    auto pad_h = get_windowed_padding(out_h, f_h, stride_h, dilation_h, options.padding() == tflite::Padding_SAME);
    auto pad_w = get_windowed_padding(out_w, f_w, stride_w, dilation_w, options.padding() == tflite::Padding_SAME);
    auto conv = graph_.emplace<conv2d_transpose>(pre_trans->output().shape(), weights_trans->output().shape(), out_shape, 1,
        pad_h, pad_w, 0, 0, stride_h, stride_w, dilation_h, dilation_w, value_range<float>::full());
    conv->name(get_tensor(op.outputs(), 0).name()->string_view());

    conv->input().connect(pre_trans->output());
    conv->weights().connect(weights_trans->output());
    if (op.inputs()->size() > 3)
        link_input_tensor(&conv->bias(), op.inputs()->Get(3));
    else
        conv->bias().connect(bias->output());

    auto sur_trans = nchw_to_nhwc(dt_float32, conv->output().shape());
    sur_trans->name(get_tensor(op.outputs(), 0).name()->string_view());
    sur_trans->input().connect(conv->output());

    auto input_conn = &pre_trans->input();
    auto output_conn = &sur_trans->output();
    [[maybe_unused]] std::vector<input_connector *> inputs_conn;
    [[maybe_unused]] std::vector<output_connector *> outputs_conn;
    [[maybe_unused]] std::vector<quant_param_t> input_dequant_params;
    [[maybe_unused]] std::vector<quant_param_t> output_quant_params;
    if (input.type() != tflite::TensorType_FLOAT32)
    {
        std::vector<input_connector *> inputs_conn = { input_conn };
        std::vector<quant_param_t> input_dequant_params = {
            quant_param_t { (int32_t)input.quantization()->zero_point()->Get(0), input.quantization()->scale()->Get(0) }
        };
        std::vector<output_connector *> outputs_conn = { output_conn };
        std::vector<quant_param_t> output_quant_params = {
            quant_param_t { (int32_t)output.quantization()->zero_point()->Get(0), output.quantization()->scale()->Get(0) }
        };
        with_quantize(to_data_type(input.type()), inputs_conn, input_dequant_params, outputs_conn, output_quant_params);
        input_conn = inputs_conn[0];
        output_conn = outputs_conn[0];
    }

    link_input_tensor(input_conn, op.inputs()->Get(2));
    link_output_tensor(op.outputs()->Get(0), output_conn);

    //weights dequantize
    if (weights.type() != tflite::TensorType_FLOAT32)
    {
        quant_param_t weights_paras = to_quant_param(weights.quantization());
        auto weights_dequant = graph_.emplace<dequantize>(to_data_type(weights.type()), get_shape(weights.shape()), dt_float32, weights_paras);
        weights_dequant->name(get_tensor(op.outputs(), 0).name()->string_view());
        //        weights_trans = nhwc_to_nchw(weights_dequant->output().type(), weights_dequant->output().shape());
        //        weights_trans->name(get_tensor(op.outputs(), 0).name()->string_view());
        weights_trans->input().connect(weights_dequant->output());
        link_input_tensor(&weights_dequant->input(), op.inputs()->Get(1));
    }
    else
    {
        //        weights_trans = graph_.emplace<transpose>(to_data_type(weights.type()), get_shape(weights.shape()), axis_t { 0, 3, 1, 2 });
        //        weights_trans->name(get_tensor(op.outputs(), 0).name()->string_view());
        link_input_tensor(&weights_trans->input(), op.inputs()->Get(1));
    }
    //    link_input_tensor(&weights_trans->input(), op.inputs()->Get(1));
}
