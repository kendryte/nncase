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
// #include <nncase/ir/ops/binary.h>
// #include <nncase/ir/ops/constant.h>
// #include <nncase/ir/ops/unary.h>
#include <nncase/ir/ops/batchnorm.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_CAFFE_LOWER(BatchNorm)
{
    // auto &input = *output_tensors_.at(op.bottom(0));
    // auto &param = op.batch_norm_param();
    // if (param.has_use_global_stats() && !param.use_global_stats())
    //     throw std::runtime_error("use_global_stats should be true at inference step");

    // auto means = load_tensor<1>(op.blobs(0));
    // auto variants = load_tensor<1>(op.blobs(1));
    // auto eps = load_tensor<1>(op.blobs(2));

    // std::vector<float> means_vec_c(means.begin(), means.end());
    // std::vector<float> variants_vec_c(variants.begin(), variants.end());
    // std::vector<float> eps_vec_c(eps.begin(), eps.end());
    // std::vector<float> means_vec;
    // std::vector<float> variants_vec;
    // std::vector<float> eps_vec;
    // for (size_t n = 0; n < input.shape()[0]; n++)
    // {
    //     for (size_t c = 0; c < input.shape()[1]; c++)
    //     {
    //         for (size_t hw = 0; hw < input.shape()[2] * input.shape()[3]; hw++)
    //         {
    //             means_vec.push_back(means_vec_c[c]);
    //             variants_vec.push_back(means_vec_c[c]);
    //             eps_vec.push_back(means_vec_c[c]);
    //         }
    //     }
    // }

    // auto means_const = graph_.emplace<constant>(dt_float32, input.shape(), means_vec);
    // means_const->name(op.name() + "/means_const");
    // auto variants_const = graph_.emplace<constant>(dt_float32, input.shape(), variants_vec);
    // variants_const->name(op.name() + "/variants_const");
    // auto eps_const = graph_.emplace<constant>(dt_float32, input.shape(), eps_vec);
    // eps_const->name(op.name() + "/eps_const");

    // auto sub = graph_.emplace<binary>(binary_sub, input.shape(), means_const->output().shape(), value_range<float>::full());
    // sub->name(op.name() + "/sub");
    // auto add = graph_.emplace<binary>(binary_add, variants_const->output().shape(), eps_const->output().shape(), value_range<float>::full());
    // add->name(op.name() + "/add");
    // auto sqrt = graph_.emplace<unary>(unary_sqrt, add->output().shape());
    // sqrt->name(op.name() + "/sqrt");
    // auto div = graph_.emplace<binary>(binary_div, sub->output().shape(), sqrt->output().shape(), value_range<float>::full());
    // div->name(op.name() + "/div");

    // sub->input_b().connect(means_const->output());
    // add->input_a().connect(variants_const->output());
    // add->input_b().connect(eps_const->output());
    // sqrt->input().connect(add->output());
    // div->input_a().connect(sub->output());
    // div->input_b().connect(sqrt->output());

    // input_tensors_.emplace(&sub->input_a(), op.bottom(0));
    // output_tensors_.emplace(op.top(0), &div->output());

    auto &input = *output_tensors_.at(op.bottom(0));
    auto &param = op.batch_norm_param();
    if (param.has_use_global_stats() && !param.use_global_stats())
        throw std::runtime_error("use_global_stats should be true at inference step");

    auto means = load_tensor<1>(op.blobs(0));
    auto variants = load_tensor<1>(op.blobs(1));
    auto eps = load_tensor<1>(op.blobs(2));
    std::vector<float> means_vec_c(means.begin(), means.end());
    std::vector<float> variants_vec_c(variants.begin(), variants.end());
    std::vector<float> eps_vec_c(eps.begin(), eps.end());

    auto bn = graph_.emplace<batchnorm>(dt_float32, input.shape(), means_vec_c, variants_vec_c, eps_vec_c);
    bn->name(op.name() + "/batchnorm");

    input_tensors_.emplace(&bn->input(), op.bottom(0));
    output_tensors_.emplace(op.top(0), &bn->output());
}
