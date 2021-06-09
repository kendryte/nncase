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
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/constant.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_CAFFE_LOWER(Scale)
{
    auto &input = *output_tensors_.at(op.bottom(0));
    auto &param = op.scale_param();

    auto gamma = load_tensor<1>(op.blobs(0));
    
    std::vector<float> gamma_vec(gamma.begin(), gamma.end());
    auto gamma_const = graph_.emplace<constant>(dt_float32, shape_t {1, get_shape(op.blobs(0).shape())[0], 1, 1}, gamma_vec);
    auto mul = graph_.emplace<binary>(binary_mul, input.shape(), gamma_const->output().shape(), value_range<float>::full());

    mul->input_b().connect(gamma_const->output());

    if (!param.has_bias_term())
    {
        input_tensors_.emplace(&mul->input_a(), op.bottom(0));
        output_tensors_.emplace(op.top(0), &mul->output());
    }
    else
    {
        auto beta = load_tensor<1>(op.blobs(1));
        std::vector<float> beta_vec(beta.begin(), beta.end());
        auto beta_const = graph_.emplace<constant>(dt_float32, shape_t {1, get_shape(op.blobs(1).shape())[0], 1, 1}, beta_vec);
        auto add = graph_.emplace<binary>(binary_add, mul->output().shape(), beta_const->output().shape(), value_range<float>::full());
        input_tensors_.emplace(&mul->input_a(), op.bottom(0));
        output_tensors_.emplace(op.top(0), &add->output());
    }
}