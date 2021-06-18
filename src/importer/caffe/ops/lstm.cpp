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
#include <nncase/ir/ops/lstm.h>
#include <nncase/ir/ops/bitcast.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace caffe;

DEFINE_CAFFE_LOWER(LSTM)
{
    // check if there are bn/scale/relu above
    std::string input_name_a = get_real_input_names(op)[0];

    auto &input_a = *output_tensors_.at(input_name_a);
    auto input_b_shape = shape_t { 1, 1, 1, 1 };
    bool has_static = false;

    auto &param = op.recurrent_param();
    auto n_output = param.num_output();

    auto op_data = get_op_data(op, caffemodel);

    // blob0: xc, blob1: static, blob2: hc
    auto blob0 = load_tensor<2>(op_data.blobs(0));
    auto blob1 = load_tensor<1>(op_data.blobs(1));
    auto blob2 = load_tensor<2>(op_data.blobs(2));

    if (op.bottom_size() == 3)
    {
        // check if there are bn/scale/relu above
        std::string input_name_b = get_real_input_names(op)[2];
        auto &input_b = *output_tensors_.at(op.bottom(2));
        input_b_shape = input_b.shape();
        has_static = true;
    }

    std::vector<float> blob0_vec(blob0.begin(), blob0.end());
    std::vector<float> blob1_vec(blob1.begin(), blob1.end());
    std::vector<float> blob2_vec(blob2.begin(), blob2.end());

    if (input_a.shape().size() != 3)
    {
        auto rshape = graph_.emplace<bitcast>(dt_float32, input_a.shape(), dt_float32, axis_t { (int32_t)input_a.shape()[0], (int32_t)input_a.shape()[1] / (int32_t)param.num_output(), (int32_t)param.num_output() });
        auto node = graph_.emplace<lstm>(rshape->output().shape(), input_b_shape, blob0_vec, blob1_vec, blob2_vec, n_output, has_static);
        node->name(op.name() + "/lstm");
        input_tensors_.emplace(&rshape->input(), input_name_a);
        node->input_a().connect(rshape->output());
        if (has_static)
        {
            // check if there are bn/scale/relu above
            std::string input_name_b = get_real_input_names(op)[2];
            input_tensors_.emplace(&node->input_b(), input_name_b);
        }
        output_tensors_.emplace(op.top(0), &node->output());
    }
    else
    {
        auto node = graph_.emplace<lstm>(input_a.shape(), input_b_shape, blob0_vec, blob1_vec, blob2_vec, n_output, has_static);
        node->name(op.name() + "/lstm");
        input_tensors_.emplace(&node->input_a(), input_name_a);
        if (has_static)
        {
            // check if there are bn/scale/relu above
            std::string input_name_b = get_real_input_names(op)[2];
            input_tensors_.emplace(&node->input_b(), input_name_b);
        }
        output_tensors_.emplace(op.top(0), &node->output());
    }
}