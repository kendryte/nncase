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
#include <nncase/ir/ops/reduce_window2d.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(AVERAGE_POOL_2D)
{
    convert_pool2d(op, reduce_mean, 0.f);
}

DEFINE_TFLITE_LOWER(MAX_POOL_2D)
{
    convert_pool2d(op, reduce_max, std::numeric_limits<float>::lowest());
}

void tflite_importer::convert_pool2d(const tflite::Operator &op, reduce_op_t reduce_op, float init_value)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &output = get_tensor(op.outputs(), 0);
    auto &options = *op.builtin_options_as_Pool2DOptions();
    dequantize *in_quant;
    quantize *out_quant;
    transpose *pre_trans;

    // input dequantize
    if (input.type() != tflite::TensorType_FLOAT32)
    {
        quant_param_t in_quant_paras = to_quant_param(input.quantization());
        in_quant = graph_.emplace<dequantize>(to_data_type(input.type()), get_shape(input.shape()), dt_float32, in_quant_paras);
        in_quant->name(get_tensor(op.outputs(), 0).name()->string_view());
        pre_trans = nhwc_to_nchw(in_quant->output().type(), in_quant->output().shape());
        pre_trans->name(get_tensor(op.outputs(), 0).name()->string_view());
        pre_trans->input().connect(in_quant->output());
        link_input_tensor(&in_quant->input(), op.inputs()->Get(0));
    }
    else
    {
        // Do not need to dequantize: Type == float32
        pre_trans = nhwc_to_nchw(to_data_type(input.type()), get_shape(input.shape()));
        pre_trans->name(get_tensor(op.outputs(), 0).name()->string_view());
        link_input_tensor(&pre_trans->input(), op.inputs()->Get(0));
    }

    auto in_h = (int32_t)pre_trans->output().shape()[2];
    auto in_w = (int32_t)pre_trans->output().shape()[3];
    auto f_h = options.filter_height();
    auto f_w = options.filter_width();
    auto stride_h = options.stride_h();
    auto stride_w = options.stride_w();
    auto dilation_h = 1;
    auto dilation_w = 1;
    auto pad_h = get_windowed_padding(in_h, f_h, stride_h, dilation_h, options.padding() == tflite::Padding_SAME);
    auto pad_w = get_windowed_padding(in_w, f_w, stride_w, dilation_w, options.padding() == tflite::Padding_SAME);
    auto conv = graph_.emplace<reduce_window2d>(reduce_op, pre_trans->output().shape(), init_value, f_h, f_w, pad_h, pad_w,
        stride_h, stride_w, dilation_h, dilation_w, to_float_clamp_range(options.fused_activation_function()));
    conv->name(get_tensor(op.outputs(), 0).name()->string_view());

    conv->input().connect(pre_trans->output());

    auto sur_trans = nchw_to_nhwc(conv->output().type(), conv->output().shape());
    sur_trans->name(get_tensor(op.outputs(), 0).name()->string_view());
    sur_trans->input().connect(conv->output());

    if (sur_trans->output().type() != to_data_type(input.type()))
    {
        quant_param_t out_quant_paras = to_quant_param(output.quantization());
        out_quant = graph_.emplace<quantize>(dt_float32, sur_trans->output().shape(), to_data_type(output.type()), out_quant_paras);
        out_quant->name(get_tensor(op.outputs(), 0).name()->string_view());
        out_quant->input().connect(sur_trans->output());
        link_output_tensor(op.outputs()->Get(0), &out_quant->output());
    }
    else
    {
        link_output_tensor(op.outputs()->Get(0), &sur_trans->output());
    }
}
