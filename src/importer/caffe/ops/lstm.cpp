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
    if (w_xc_shape.size() == 2)
    {
        w_xc_shape = shape_t { 1, w_xc_shape[0], w_xc_shape[1] };
    }

    if (op_data.blobs_size() == 4)
    {
        has_static = true;
        auto blob_w_static = load_tensor<1>(op_data.blobs(2));
        static_shape = get_shape(op_data.blobs(2).shape());
        blob_w_static_vec.assign(blob_w_static.begin(), blob_w_static.end());
    }
    auto blob_w_rc = op_data.blobs_size() == 4 ? load_tensor<2>(op_data.blobs(3)) : load_tensor<2>(op_data.blobs(2));
    auto w_rc_shape = op_data.blobs_size() == 4 ? get_shape(op_data.blobs(3).shape()) : get_shape(op_data.blobs(2).shape());
    auto b_rc_shape = b_xc_shape;

    if (w_rc_shape.size() == 2)
    {
        w_rc_shape = shape_t { 1, w_rc_shape[0], w_rc_shape[1] };
    }

    std::vector<float> blob_w_xc_vec(blob_w_xc.begin(), blob_w_xc.end());
    std::vector<float> blob_b_xc_vec(blob_b_xc.begin(), blob_b_xc.end());
    std::vector<float> blob_w_rc_vec(blob_w_rc.begin(), blob_w_rc.end());
    std::vector<float> blob_b_rc_vec((int)b_rc_shape[0], 0.f);

    // create init_h init_c
    std::vector<float> init_const(w_rc_shape[2], 0.f);
    auto init_h = graph_.emplace<constant>(dt_float32, shape_t { 1, 1, w_rc_shape[2] }, init_const);
    auto init_c = graph_.emplace<constant>(dt_float32, shape_t { 1, 1, w_rc_shape[2] }, init_const);
    init_h->name(op.name() + "init_h");
    init_c->name(op.name() + "init_c");

    if (input.shape().size() != 3)
    {
        auto rshape = graph_.emplace<bitcast>(dt_float32, input.shape(), dt_float32, axis_t { (int32_t)input.shape()[0], (int32_t)input.shape()[1] / (int32_t)param.num_output(), (int32_t)param.num_output() });
        auto node = graph_.emplace<lstm>(rshape->output().shape(), w_xc_shape, b_xc_shape, w_rc_shape, b_rc_shape, init_h->output().shape(), init_c->output().shape(), n_output, has_static, "caffe");
        node->name(op.name() + "/lstm");
        input_tensors_.emplace(&rshape->input(), input_name);
        node->input().connect(rshape->output());

        auto w_xc_const = graph_.emplace<constant>(dt_float32, w_xc_shape, blob_w_xc_vec);
        auto b_xc_const = graph_.emplace<constant>(dt_float32, b_xc_shape, blob_b_xc_vec);
        auto w_rc_const = graph_.emplace<constant>(dt_float32, w_rc_shape, blob_w_rc_vec);
        auto b_rc_const = graph_.emplace<constant>(dt_float32, b_rc_shape, blob_b_rc_vec);

        w_xc_const->name(op.name() + "/w_xc_const");
        b_xc_const->name(op.name() + "/b_xc_const");
        w_rc_const->name(op.name() + "/w_rc_const");
        b_rc_const->name(op.name() + "/b_rc_const");

        node->w_xc().connect(w_xc_const->output());
        node->b_xc().connect(b_xc_const->output());
        node->w_rc().connect(w_rc_const->output());
        node->b_rc().connect(b_rc_const->output());
        node->initial_h().connect(init_h->output());
        node->initial_c().connect(init_c->output());

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
        auto node = graph_.emplace<lstm>(input.shape(), w_xc_shape, b_xc_shape, w_rc_shape, b_rc_shape, init_h->output().shape(), init_c->output().shape(), n_output, has_static, "caffe");
        node->name(op.name() + "/lstm");
        input_tensors_.emplace(&node->input(), input_name);

        auto w_xc_const = graph_.emplace<constant>(dt_float32, w_xc_shape, blob_w_xc_vec);
        auto b_xc_const = graph_.emplace<constant>(dt_float32, b_xc_shape, blob_b_xc_vec);
        auto w_rc_const = graph_.emplace<constant>(dt_float32, w_rc_shape, blob_w_rc_vec);
        auto b_rc_const = graph_.emplace<constant>(dt_float32, b_rc_shape, blob_b_rc_vec);

        w_xc_const->name(op.name() + "/w_xc_const");
        b_xc_const->name(op.name() + "/b_xc_const");
        w_rc_const->name(op.name() + "/w_rc_const");
        b_rc_const->name(op.name() + "/b_rc_const");

        node->w_xc().connect(w_xc_const->output());
        node->b_xc().connect(b_xc_const->output());
        node->w_rc().connect(w_rc_const->output());
        node->b_rc().connect(b_rc_const->output());
        node->initial_h().connect(init_h->output());
        node->initial_c().connect(init_c->output());

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
