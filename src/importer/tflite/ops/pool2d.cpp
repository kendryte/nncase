/* Copyright 2019 Canaan Inc.
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
#include <ir/ops/reduce_window2d.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(AVERAGE_POOL_2D)
{
    convert_pool_2d(op, reduce_mean, 0.f);
}

DEFINE_TFLITE_LOWER(MAX_POOL_2D)
{
    convert_pool_2d(op, reduce_max, std::numeric_limits<float>::lowest());
}

void tflite_importer::convert_pool_2d(const tflite::Operator &op, reduce_op_t reduce_op, float init_value)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &options = *op.builtin_options_as_Pool2DOptions();

    auto pre_trans = nhwc_to_nchw(dt_float32, get_shape(input.shape()));

    auto in_h = pre_trans->output().shape()[2];
    auto in_w = pre_trans->output().shape()[3];
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
    conv->input().connect(pre_trans->output());

    auto sur_trans = nchw_to_nhwc(dt_float32, conv->output().shape());
    sur_trans->input().connect(conv->output());

    input_tensors_.emplace(&pre_trans->input(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &sur_trans->output());
}
