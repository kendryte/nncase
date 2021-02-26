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
#include <nncase/ir/ops/conv2d.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(CONV_2D)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &weights = get_tensor(op.inputs(), 1);
    auto &bias = get_tensor(op.inputs(), 2);
    auto &output = get_tensor(op.outputs(), 0);

    auto &options = *op.builtin_options_as_Conv2DOptions();

    dequantize *in_dequant, *weights_dequant, *bias_dequant;
    transpose *pre_trans, *weights_trans;

    // data dequantize
    if (input.type() != tflite::TensorType_FLOAT32)
    {
        quant_param_t in_deq_params;
        in_deq_params.scale = to_vector(*input.quantization()->scale());
        in_deq_params.zero_point = to_vector(*input.quantization()->zero_point());

        in_dequant = graph_.emplace<dequantize>(to_data_type(input.type()), get_shape(input.shape()), dt_float32, in_deq_params);
        in_dequant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/in_deq");
        pre_trans = nhwc_to_nchw(in_dequant->output().type(), in_dequant->output().shape());
        pre_trans->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/in_tp");
        pre_trans->input().connect(in_dequant->output());
        link_input_tensor(&in_dequant->input(), op.inputs()->Get(0));
    }
    else
    {
        pre_trans = nhwc_to_nchw(to_data_type(input.type()), get_shape(input.shape()));
        pre_trans->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/in_tp");
        link_input_tensor(&pre_trans->input(), op.inputs()->Get(0));
    }

    //weights dequantize
    if (weights.type() != tflite::TensorType_FLOAT32)
    {
        quant_param_t weights_paras;
        weights_paras.scale = to_vector(*weights.quantization()->scale());
        weights_paras.zero_point = to_vector(*weights.quantization()->zero_point());

        weights_dequant = graph_.emplace<dequantize>(to_data_type(weights.type()), get_shape(weights.shape()), dt_float32, weights_paras);
        weights_dequant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/weights_deq");
        weights_trans = nhwc_to_nchw(weights_dequant->output().type(), weights_dequant->output().shape());
        weights_trans->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/weights_tp");
        weights_trans->input().connect(weights_dequant->output());
        link_input_tensor(&weights_dequant->input(), op.inputs()->Get(1));
    }
    else
    {
        weights_trans = graph_.emplace<transpose>(to_data_type(weights.type()), get_shape(weights.shape()), axis_t { 0, 3, 1, 2 });
        weights_trans->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/weights_tp");
        link_input_tensor(&weights_trans->input(), op.inputs()->Get(1));
    }

    auto in_h = (int32_t)pre_trans->output().shape()[2];
    auto in_w = (int32_t)pre_trans->output().shape()[3];
    auto f_h = (int32_t)weights_trans->output().shape()[2];
    auto f_w = (int32_t)weights_trans->output().shape()[3];
    auto stride_h = options.stride_h();
    auto stride_w = options.stride_w();
    auto dilation_h = options.dilation_h_factor();
    auto dilation_w = options.dilation_w_factor();
    auto pad_h = get_windowed_padding(in_h, f_h, stride_h, dilation_h, options.padding() == tflite::Padding_SAME);
    auto pad_w = get_windowed_padding(in_w, f_w, stride_w, dilation_w, options.padding() == tflite::Padding_SAME);
    auto clamp = to_float_clamp_range(options.fused_activation_function());
    if (output.type() != tflite::TensorType_FLOAT32)
    {
        auto quant_max = to_vector(*output.quantization()->max());
        auto quant_min = to_vector(*output.quantization()->min());
        if (clamp.max > quant_max[0])
        {
            clamp.max = quant_max[0];
        }
        if (clamp.min < quant_min[0])
        {
            clamp.min = quant_min[0];
        }
    }
    auto conv = graph_.emplace<conv2d>(pre_trans->output().shape(), weights_trans->output().shape(), 1, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, clamp);
    conv->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/conv");

    conv->input().connect(pre_trans->output());
    conv->weights().connect(weights_trans->output());

    //bias dequant
    if (bias.type() != tflite::TensorType_FLOAT32)
    {
        quant_param_t bias_deq_params;
        bias_deq_params.scale = to_vector(*bias.quantization()->scale());
        bias_deq_params.zero_point = to_vector(*bias.quantization()->zero_point());

        bias_dequant = graph_.emplace<dequantize>(to_data_type(bias.type()), get_shape(bias.shape()), dt_float32, bias_deq_params);
        bias_dequant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/bias_dequant");
        conv->bias().connect(bias_dequant->output());
        link_input_tensor(&bias_dequant->input(), op.inputs()->Get(2));
    }
    else
    {
        link_input_tensor(&conv->bias(), op.inputs()->Get(2));
    }

    auto sur_trans = nchw_to_nhwc(conv->output().type(), conv->output().shape());
    sur_trans->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/out_tp");

    sur_trans->input().connect(conv->output());

    if (sur_trans->output().type() != to_data_type(input.type()))
    {
        quant_param_t out_quant;
        out_quant.scale = to_vector(*output.quantization()->scale());
        out_quant.zero_point = to_vector(*output.quantization()->zero_point());
        auto data_quantize = graph_.emplace<quantize>(dt_float32, sur_trans->output().shape(), to_data_type(output.type()), out_quant);
        data_quantize->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/out_quant");
        data_quantize->input().connect(sur_trans->output());

        link_output_tensor(op.outputs()->Get(0), &data_quantize->output());
    }
    else
    {
        link_output_tensor(op.outputs()->Get(0), &sur_trans->output());
    }
}

DEFINE_TFLITE_LOWER(DEPTHWISE_CONV_2D)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &weights = get_tensor(op.inputs(), 1);
    auto &bias = get_tensor(op.inputs(), 2);
    auto &output = get_tensor(op.outputs(), 0);

    auto &options = *op.builtin_options_as_DepthwiseConv2DOptions();
    auto opname = weights.name()->string_view().substr(0, weights.name()->string_view().find_first_of('/'));

    dequantize *data_dequant, *weights_dequant, *bias_dequant;
    transpose *pre_trans, *weights_trans;
    quantize *data_quant;
    // data dequantize
    if (input.type() != tflite::TensorType_FLOAT32)
    {
        quant_param_t data_paras;
        data_paras.scale = to_vector(*input.quantization()->scale());
        data_paras.zero_point = to_vector(*input.quantization()->zero_point());

        data_dequant = graph_.emplace<dequantize>(to_data_type(input.type()), get_shape(input.shape()), dt_float32, data_paras);
        data_dequant->name(get_tensor(op.outputs(), 0).name()->string_view());
        pre_trans = nhwc_to_nchw(data_dequant->output().type(), data_dequant->output().shape());
        pre_trans->input().connect(data_dequant->output());
        link_input_tensor(&data_dequant->input(), op.inputs()->Get(0));
    }
    else
    {
        pre_trans = nhwc_to_nchw(to_data_type(input.type()), get_shape(input.shape()));
        link_input_tensor(&pre_trans->input(), op.inputs()->Get(0));
    }

    //weights dequantize
    if (weights.type() != tflite::TensorType_FLOAT32)
    {
        quant_param_t weights_paras;
        weights_paras.scale = to_vector(*weights.quantization()->scale());
        weights_paras.zero_point = to_vector(*weights.quantization()->zero_point());

        weights_dequant = graph_.emplace<dequantize>(to_data_type(weights.type()), get_shape(weights.shape()), dt_float32, weights_paras);
        weights_dequant->name(get_tensor(op.outputs(), 0).name()->string_view());

        weights_trans = graph_.emplace<transpose>(weights_dequant->output().type(), weights_dequant->output().shape(), axis_t { 3, 0, 1, 2 });
        weights_trans->name(get_tensor(op.outputs(), 0).name()->string_view());

        weights_trans->input().connect(weights_dequant->output());
        link_input_tensor(&weights_dequant->input(), op.inputs()->Get(1));
    }
    else
    {
        weights_trans = graph_.emplace<transpose>(to_data_type(weights.type()), get_shape(weights.shape()), axis_t { 3, 0, 1, 2 });
        weights_trans->name(get_tensor(op.outputs(), 0).name()->string_view());
        link_input_tensor(&weights_trans->input(), op.inputs()->Get(1));
    }

    auto in_h = (int32_t)pre_trans->output().shape()[2];
    auto in_w = (int32_t)pre_trans->output().shape()[3];
    auto groups = (int32_t)weights_trans->output().shape()[0];
    auto f_h = (int32_t)weights_trans->output().shape()[2];
    auto f_w = (int32_t)weights_trans->output().shape()[3];
    auto stride_h = options.stride_h();
    auto stride_w = options.stride_w();
    auto dilation_h = options.dilation_h_factor();
    auto dilation_w = options.dilation_w_factor();
    auto pad_h = get_windowed_padding(in_h, f_h, stride_h, dilation_h, options.padding() == tflite::Padding_SAME);
    auto pad_w = get_windowed_padding(in_w, f_w, stride_w, dilation_w, options.padding() == tflite::Padding_SAME);
    auto depth_mul = options.depth_multiplier();
    conv2d *conv;
    auto clamp = to_float_clamp_range(options.fused_activation_function());
    if (output.type() != tflite::TensorType_FLOAT32)
    {
        auto quant_max = to_vector(*output.quantization()->max());
        auto quant_min = to_vector(*output.quantization()->min());
        if (clamp.max > quant_max[0])
        {
            clamp.max = quant_max[0];
        }
        if (clamp.min < quant_min[0])
        {
            clamp.min = quant_min[0];
        }
    }
    if (pre_trans->output().shape()[1] == 1 && depth_mul == groups)
    {
        conv = graph_.emplace<conv2d>(pre_trans->output().shape(), weights_trans->output().shape(), 1,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, clamp);
        conv->name(get_tensor(op.outputs(), 0).name()->string_view());
    }
    else if (depth_mul != 1)
    {
        throw std::runtime_error("DepthwiseConv2d " + std::string(opname) + " with depth_multiplier " + std::to_string(depth_mul) + " is not supported");
    }
    else
    {
        conv = graph_.emplace<conv2d>(pre_trans->output().shape(), weights_trans->output().shape(), groups,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, clamp);
        conv->name(get_tensor(op.outputs(), 0).name()->string_view());
    }

    conv->input().connect(pre_trans->output());
    conv->weights().connect(weights_trans->output());

    //bias dequantize
    if (bias.type() != tflite::TensorType_FLOAT32)
    {
        quant_param_t bias_paras;
        bias_paras.scale = to_vector(*bias.quantization()->scale());
        bias_paras.zero_point = to_vector(*bias.quantization()->zero_point());

        bias_dequant = graph_.emplace<dequantize>(to_data_type(bias.type()), get_shape(bias.shape()), dt_float32, bias_paras);
        conv->bias().connect(bias_dequant->output());
        link_input_tensor(&bias_dequant->input(), op.inputs()->Get(2));
    }
    else
    {
        link_input_tensor(&conv->bias(), op.inputs()->Get(2));
    }

    auto sur_trans = nchw_to_nhwc(dt_float32, conv->output().shape());
    sur_trans->name(get_tensor(op.outputs(), 0).name()->string_view());
    sur_trans->input().connect(conv->output());
    //output quantize
    if (sur_trans->output().type() != to_data_type(input.type()))
    {
        quant_param_t quant {
            to_vector(*output.quantization()->zero_point()),
            to_vector(*output.quantization()->scale())
        };
        data_quant = graph_.emplace<quantize>(dt_float32, sur_trans->output().shape(), to_data_type(output.type()), quant);
        data_quant->input().connect(sur_trans->output());

        link_output_tensor(op.outputs()->Get(0), &data_quant->output());
    }
    else
    {
        link_output_tensor(op.outputs()->Get(0), &sur_trans->output());
    }
}
