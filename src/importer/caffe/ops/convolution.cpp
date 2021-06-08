/* Copyright 2019-2020 Canaan Inc.
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
#include "../caffe_importer.h"
#include <functional>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/constant.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace caffe;

DEFINE_CAFFE_LOWER(Convolution)
{
    using namespace std::placeholders;

    typedef uint32_t (ConvolutionParameter::*arr_func_t)(int32_t) const;

    auto &input = *output_tensors_.at(op.bottom(0));
    auto &param = op.convolution_param();
    auto pad_h = (int32_t)get_or_default(std::bind(arr_func_t(&ConvolutionParameter::pad), &param, _1), param.pad_size(), 0, param.pad_h());
    auto pad_w = (int32_t)get_or_default(std::bind(arr_func_t(&ConvolutionParameter::pad), &param, _1), param.pad_size(), 1, param.pad_w());
    auto groups = param.group();
    auto stride_h = get_or_default(std::bind(arr_func_t(&ConvolutionParameter::stride), &param, _1), param.stride_size(), 0, 1);
    auto stride_w = get_or_default(std::bind(arr_func_t(&ConvolutionParameter::stride), &param, _1), param.stride_size(), 1, 1);
    auto dilation_h = get_or_default(std::bind(arr_func_t(&ConvolutionParameter::dilation), &param, _1), param.dilation_size(), 0, 1);
    auto dilation_w = get_or_default(std::bind(arr_func_t(&ConvolutionParameter::dilation), &param, _1), param.dilation_size(), 1, 1);

    auto weights = load_tensor<4>(op.blobs(0));
    auto bias = load_tensor<1>(op.blobs(1));

    auto node = graph_.emplace<conv2d>(input.shape(), get_shape(op.blobs(0).shape()), groups, padding { pad_h, pad_h }, padding { pad_w, pad_w },
    (int32_t)stride_h, (int32_t)stride_w, (int32_t)dilation_h, (int32_t)dilation_w, value_range<float>::full());

    std::vector<float> weights_vec(weights.begin(), weights.end());
    std::vector<float> bias_vec(bias.begin(), bias.end());
    auto weights_const = graph_.emplace<constant>(dt_float32, get_shape(op.blobs(0).shape()), weights_vec);
    auto bias_const = graph_.emplace<constant>(dt_float32, get_shape(op.blobs(1).shape()), bias_vec);

    input_tensors_.emplace(&node->input(), op.bottom(0));
    node->weights().connect(weights_const->output());
    node->bias().connect(bias_const->output());
    output_tensors_.emplace(op.top(0), &node->output());
}
