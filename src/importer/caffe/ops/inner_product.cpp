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
#include <nncase/ir/ops/matmul.h>
#include <nncase/ir/ops/constant.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace caffe;

DEFINE_CAFFE_LOWER(InnerProduct)
{
    auto &input = *output_tensors_.at(op.bottom(0));
    auto &param = op.inner_product_param();

    auto input_b = load_tensor<2>(op.blobs(0));
    std::vector<float> input_b_vec(input_b.begin(), input_b.end());
    auto input_b_const = graph_.emplace<constant>(dt_float32, get_shape(op.blobs(0).shape()), input_b_vec);
    input_b_const->name(op.name() + "/input_b_const");

    auto node = graph_.emplace<matmul>(input.shape(), get_shape(op.blobs(0).shape()), value_range<float>::full());
    node->name(op.name() + "/matmul");

    input_tensors_.emplace(&node->input_a(), op.bottom(0));
    node->input_b().connect(input_b_const->output());
    if (param.has_bias_term())
    {
        auto bias = load_tensor<1>(op.blobs(1));
        std::vector<float> bias_vec(bias.begin(), bias.end());
        auto bias_const = graph_.emplace<constant>(dt_float32, get_shape(op.blobs(1).shape()), bias_vec);
        bias_const->name(op.name() + "/bias_const");
        node->bias().connect(bias_const->output());
    }
    else
    {
        std::vector<float> bias_vec(input_b.shape()[1], 0);
        auto bias_const = graph_.emplace<constant>(dt_float32, shape_t { input_b.shape()[1] }, bias_vec);
        bias_const->name(op.name() + "/bias_const");
        node->bias().connect(bias_const->output());
    }
    output_tensors_.emplace(op.top(0), &node->output());
}
