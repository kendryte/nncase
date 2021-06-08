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
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_CAFFE_LOWER(BatchNorm)
{
    auto &input = *output_tensors_.at(op.bottom(0));
    auto &param = op.batch_norm_param();
    if (param.has_use_global_stats() && !param.use_global_stats())
        throw std::runtime_error("use_global_stats should be true at inference step");

    auto means = load_tensor<1>(op.blobs(0));
    auto variants = load_tensor<1>(op.blobs(1));
    auto eps = load_tensor<1>(op.blobs(2));

    std::vector<float> means_vec(means.begin(), means.end());
    std::vector<float> variants_vec(variants.begin(), variants.end());
    std::vector<float> eps_vec(eps.begin(), eps.end());

    auto means_const = graph_.emplace<constant>(dt_float32, shape_t {1, get_shape(op.blobs(0).shape())[0], 1, 1}, means_vec);
    auto variants_const = graph_.emplace<constant>(dt_float32, shape_t {1, get_shape(op.blobs(1).shape())[0], 1, 1}, variants_vec);
    auto eps_const = graph_.emplace<constant>(dt_float32, shape_t {1, get_shape(op.blobs(2).shape())[0], 1, 1}, eps_vec);

    auto sub = graph_.emplace<binary>(binary_sub, input.shape(), means_const->output().shape(), value_range<float>::full());
    auto add = graph_.emplace<binary>(binary_add, variants_const->output().shape(), eps_const->output().shape(), value_range<float>::full());
    auto sqrt = graph_.emplace<unary>(unary_sqrt, add->output().shape());
    auto div = graph_.emplace<binary>(binary_div, sub->output().shape(), sqrt->output().shape(), value_range<float>::full());

    sub->input_b().connect(means_const->output());
    add->input_a().connect(variants_const->output());
    add->input_b().connect(eps_const->output());
    sqrt->input().connect(add->output());
    div->input_a().connect(sub->output());
    div->input_b().connect(sqrt->output());

    input_tensors_.emplace(&sub->input_a(), op.bottom(0));
    output_tensors_.emplace(op.top(0), &div->output());
}
