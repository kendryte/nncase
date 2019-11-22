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
#include <ir/ops/conv2d_transpose.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(TRANSPOSE_CONV)
{
    auto out_shape = nhwc_to_nchw(load_shape<int32_t>(get_tensor(op.inputs(), 0)));
    auto &input = get_tensor(op.inputs(), 2);
    auto &weights = get_tensor(op.inputs(), 1);
    auto &options = *op.builtin_options_as_TransposeConvOptions();

    auto weights_tensor = xt::transpose(dequantize_tensor<4>(weights), { 0, 3, 1, 2 });
    auto bias_tensor = xt::xtensor<float, 1>::from_shape({ out_shape[1] });
    bias_tensor.fill(0);

    auto pre_trans = nhwc_to_nchw(dt_float32, get_shape(input.shape()));

    auto out_h = out_shape[2];
    auto out_w = out_shape[3];
    auto f_h = weights_tensor.shape()[2];
    auto f_w = weights_tensor.shape()[3];
    auto stride_h = options.stride_h();
    auto stride_w = options.stride_w();
    auto dilation_h = 1;
    auto dilation_w = 1;
    auto pad_h = get_windowed_padding(out_h, f_h, stride_h, dilation_h, options.padding() == tflite::Padding_SAME);
    auto pad_w = get_windowed_padding(out_w, f_w, stride_w, dilation_w, options.padding() == tflite::Padding_SAME);
    auto conv = graph_.emplace<conv2d_transpose>(pre_trans->output().shape(), out_shape, std::move(weights_tensor), std::move(bias_tensor), 1,
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, value_range<float>::full());
    conv->input().connect(pre_trans->output());

    auto sur_trans = nchw_to_nhwc(dt_float32, conv->output().shape());
    sur_trans->input().connect(conv->output());

    input_tensors_.emplace(&pre_trans->input(), op.inputs()->Get(2));
    output_tensors_.emplace(op.outputs()->Get(0), &sur_trans->output());
}
