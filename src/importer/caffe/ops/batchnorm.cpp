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
#include "../caffe_importer.h"
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_CAFFE_LOWER(BatchNorm)
{
    // check if there are bn/scale/relu above
    std::string input_name = get_real_input_names(op)[0];

    auto &input = *output_tensors_.at(input_name);

    auto &param = op.batch_norm_param();
    if (param.has_use_global_stats() && !param.use_global_stats())
        throw std::runtime_error("use_global_stats should be true at inference step");

    auto op_data = get_op_data(op, caffemodel);

    auto means = load_tensor<1>(op_data.blobs(0));
    auto variants = load_tensor<1>(op_data.blobs(1));
    auto eps = param.eps();
    auto scale_factor = load_tensor<1>(op_data.blobs(2));

    std::vector<float> means_vec_c(means.begin(), means.end());
    std::vector<float> variants_vec_c(variants.begin(), variants.end());
    std::vector<float> scale_factor_vec(scale_factor.begin(), scale_factor.end());
    for (size_t idx = 0; idx < means_vec_c.size(); idx++)
    {
        means_vec_c[idx] /= scale_factor_vec[0];
        variants_vec_c[idx] /= scale_factor_vec[0];
    }

    std::vector<float> eps_vec(1, eps);
    auto means_const = graph_.emplace<constant>(dt_float32, shape_t { 1, input.shape()[1], 1, 1 }, means_vec_c);
    means_const->name(op.name() + "/means_const");
    auto variants_const = graph_.emplace<constant>(dt_float32, shape_t { 1, input.shape()[1], 1, 1 }, variants_vec_c);
    variants_const->name(op.name() + "/variants_const");
    auto eps_const = graph_.emplace<constant>(dt_float32, shape_t { 1, 1, 1, 1 }, eps_vec);
    eps_const->name(op.name() + "/eps_const");

    auto sub = graph_.emplace<binary>(binary_sub, dt_float32, input.shape(), means_const->output().shape(), value_range<float>::full());
    sub->name(op.name() + "/sub");
    auto add = graph_.emplace<binary>(binary_add, dt_float32, variants_const->output().shape(), eps_const->output().shape(), value_range<float>::full());
    add->name(op.name() + "/add");
    auto sqrt = graph_.emplace<unary>(unary_sqrt, add->output().shape());
    sqrt->name(op.name() + "/sqrt");
    auto div = graph_.emplace<binary>(binary_div, dt_float32, sub->output().shape(), sqrt->output().shape(), value_range<float>::full());
    if (op.bottom(0) == op.top(0))
    {
        // inplace op, user op need this name
        div->name(op.top(0) + "/div");
    }
    else
        div->name(op.name() + "/div");

    sub->input_b().connect(means_const->output());
    add->input_a().connect(variants_const->output());
    add->input_b().connect(eps_const->output());
    sqrt->input().connect(add->output());
    div->input_a().connect(sub->output());
    div->input_b().connect(sqrt->output());

    input_tensors_.emplace(&sub->input_a(), op.bottom(0));
    if (op.bottom(0) == op.top(0))
    {
        // inplace op, user op need this name
        output_tensors_.emplace(div->name(), &div->output());
    }
    else
        output_tensors_.emplace(op.top(0), &div->output());
}
