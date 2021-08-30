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
#include <functional>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/matmul.h>
#include <nncase/ir/ops/transpose.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace caffe;

DEFINE_CAFFE_LOWER(InnerProduct)
{
    // check if there are bn/scale/relu above
    std::string input_name = get_real_input_names(op)[0];
    auto &input = *output_tensors_.at(input_name);
    auto &param = op.inner_product_param();

    auto op_data = get_op_data(op, caffemodel);

    auto input_b = load_tensor<2>(op_data.blobs(0));
    std::vector<float> input_b_vec(input_b.begin(), input_b.end());
    auto input_b_const = graph_.emplace<constant>(dt_float32, get_shape(op_data.blobs(0).shape()), input_b_vec);
    input_b_const->name(op.name() + "/input_b_const");

    axis_t weights_axis = param.transpose() ? axis_t { 0, 1 } : axis_t { 1, 0 };
    auto tp_pre = graph_.emplace<nncase::ir::transpose>(dt_float32, get_shape(op_data.blobs(0).shape()), weights_axis);

    auto normalized_axis = param.axis() >= 0 ? param.axis() : (input.shape().size() + param.axis());
    size_t flattend_shape_a = 1;
    size_t flattend_shape_b = 1;
    for (size_t i = 0; i < normalized_axis; i++)
    {
        flattend_shape_a *= input.shape()[i];
    }
    for (size_t i = normalized_axis; i < input.shape().size(); i++)
    {
        flattend_shape_b *= input.shape()[i];
    }
    auto bc_pre = graph_.emplace<bitcast>(dt_float32, input.shape(), dt_float32, shape_t { flattend_shape_a, flattend_shape_b });
    auto node = graph_.emplace<matmul>(bc_pre->output().shape(), tp_pre->output().shape(), value_range<float>::full());
    shape_t bc_post_shape;
    for (size_t i = 0; i < normalized_axis; i++)
    {
        bc_post_shape.push_back(input.shape()[i]);
    }
    bc_post_shape.push_back(param.num_output());
    auto bc_post = graph_.emplace<bitcast>(dt_float32, node->output().shape(), dt_float32, bc_post_shape);
    node->name(op.name() + "/matmul");

    input_tensors_.emplace(&bc_pre->input(), input_name);
    node->input_a().connect(bc_pre->output());
    tp_pre->input().connect(input_b_const->output());
    node->input_b().connect(tp_pre->output());
    bc_post->input().connect(node->output());
    if (param.bias_term())
    {
        auto bias = load_tensor<1>(op_data.blobs(1));
        std::vector<float> bias_vec(bias.begin(), bias.end());
        auto bias_const = graph_.emplace<constant>(dt_float32, get_shape(op_data.blobs(1).shape()), bias_vec);
        bias_const->name(op.name() + "/bias_const");
        node->bias().connect(bias_const->output());
    }
    else
    {
        std::vector<float> bias_vec(tp_pre->output().shape()[1], 0);
        auto bias_const = graph_.emplace<constant>(dt_float32, shape_t { bias_vec.size() }, bias_vec);
        bias_const->name(op.name() + "/bias_const");
        node->bias().connect(bias_const->output());
    }
    output_tensors_.emplace(op.top(0), &bc_post->output());
}
