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
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/lstm.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace caffe;

DEFINE_CAFFE_LOWER(LSTM)
{
    // check if there are bn/scale/relu above
    std::string input_name = get_real_input_names(op)[0];

    auto &input = *output_tensors_.at(input_name);
    auto static_shape = shape_t { 1, 1, 1, 1 };
    bool has_static = false;

    auto &param = op.recurrent_param();
    auto n_output = param.num_output();

    if (param.expose_hidden())
        throw std::runtime_error("expose hidden for lstm is not supported yet");

    auto op_data = get_op_data(op, caffemodel);

    std::vector<float> blob_w_static_vec;

    auto blob_w_xc = load_tensor<2>(op_data.blobs(0));
    auto blob_b_xc = load_tensor<1>(op_data.blobs(1));
    auto w_xc_shape = get_shape(op_data.blobs(0).shape());
    auto b_xc_shape = get_shape(op_data.blobs(1).shape());
    if (op_data.blobs_size() == 4)
    {
        has_static = true;
        auto blob_w_static = load_tensor<1>(op_data.blobs(2));
        static_shape = get_shape(op_data.blobs(2).shape());
        blob_w_static_vec.assign(blob_w_static.begin(), blob_w_static.end());
    }
    auto blob_w_hc = op_data.blobs_size() == 4 ? load_tensor<2>(op_data.blobs(3)) : load_tensor<2>(op_data.blobs(2));
    auto w_hc_shape = op_data.blobs_size() == 4 ? get_shape(op_data.blobs(3).shape()) : get_shape(op_data.blobs(2).shape());

    std::vector<float> blob_w_xc_vec(blob_w_xc.begin(), blob_w_xc.end());
    std::vector<float> blob_b_xc_vec(blob_b_xc.begin(), blob_b_xc.end());
    std::vector<float> blob_w_hc_vec(blob_w_hc.begin(), blob_w_hc.end());

    if (input.shape().size() != 3)
    {
        auto rshape = graph_.emplace<bitcast>(dt_float32, input.shape(), dt_float32, axis_t { (int32_t)input.shape()[0], (int32_t)input.shape()[1] / (int32_t)param.num_output(), (int32_t)param.num_output() });
        auto node = graph_.emplace<lstm>(rshape->output().shape(), w_xc_shape, b_xc_shape, w_hc_shape, n_output, has_static);
        node->name(op.name() + "/lstm");
        input_tensors_.emplace(&rshape->input(), input_name);
        node->input().connect(rshape->output());

        auto w_xc_const = graph_.emplace<constant>(dt_float32, w_xc_shape, blob_w_xc_vec);
        auto b_xc_const = graph_.emplace<constant>(dt_float32, b_xc_shape, blob_b_xc_vec);
        auto w_hc_const = graph_.emplace<constant>(dt_float32, w_hc_shape, blob_w_hc_vec);

        w_xc_const->name(op.name() + "/w_xc_const");
        b_xc_const->name(op.name() + "/b_xc_const");
        w_hc_const->name(op.name() + "/w_hc_const");

        node->w_xc().connect(w_xc_const->output());
        node->b_xc().connect(b_xc_const->output());
        node->w_hc().connect(w_hc_const->output());

        if (has_static)
        {
            auto w_static_const = graph_.emplace<constant>(dt_float32, static_shape, blob_w_static_vec);
            w_static_const->name(op.name() + "/w_static_const");

            // check if there are bn/scale/relu above
            std::string static_name = get_real_input_names(op)[2];
            input_tensors_.emplace(&node->w_static(), static_name);
            node->w_static().connect(w_static_const->output());
        }
        output_tensors_.emplace(op.top(0), &node->output());
    }
    else
    {
        auto node = graph_.emplace<lstm>(input.shape(), w_xc_shape, b_xc_shape, w_hc_shape, n_output, has_static);
        node->name(op.name() + "/lstm");
        input_tensors_.emplace(&node->input(), input_name);

        auto w_xc_const = graph_.emplace<constant>(dt_float32, w_xc_shape, blob_w_xc_vec);
        auto b_xc_const = graph_.emplace<constant>(dt_float32, b_xc_shape, blob_b_xc_vec);
        auto w_hc_const = graph_.emplace<constant>(dt_float32, w_hc_shape, blob_w_hc_vec);

        w_xc_const->name(op.name() + "/w_xc_const");
        b_xc_const->name(op.name() + "/b_xc_const");
        w_hc_const->name(op.name() + "/w_hc_const");

        node->w_xc().connect(w_xc_const->output());
        node->b_xc().connect(b_xc_const->output());
        node->w_hc().connect(w_hc_const->output());

        if (has_static)
        {
            auto w_static_const = graph_.emplace<constant>(dt_float32, static_shape, blob_w_static_vec);
            w_static_const->name(op.name() + "/w_static_const");

            // check if there are bn/scale/relu above
            std::string static_name = get_real_input_names(op)[2];
            input_tensors_.emplace(&node->w_static(), static_name);
            node->w_static().connect(w_static_const->output());
        }
        output_tensors_.emplace(op.top(0), &node->output());
    }
}
