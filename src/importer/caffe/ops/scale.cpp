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

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_CAFFE_LOWER(Scale)
{
    if (op.bottom_size() == 1)
    {
        // check if there are bn/scale/relu above
        std::string input_name = get_real_input_names(op)[0];

        auto &input = *output_tensors_.at(input_name);
        auto &param = op.scale_param();

        auto op_data = get_op_data(op, caffemodel);

        auto gamma = load_tensor<1>(op_data.blobs(0));

        std::vector<float> gamma_vec_c(gamma.begin(), gamma.end());

        auto gamma_const = graph_.emplace<constant>(dt_float32, shape_t { 1, input.shape()[1], 1, 1 }, gamma_vec_c);
        gamma_const->name(op.name() + "/gamma_const");
        auto mul = graph_.emplace<binary>(binary_mul, dt_float32, input.shape(), gamma_const->output().shape(), value_range<float>::full());

        mul->input_b().connect(gamma_const->output());

        if (!param.bias_term())
        {
            if (op.bottom(0) == op.top(0))
            {
                // inplace op, user op need this name
                mul->name(op.top(0) + "/mul");
            }
            else
                mul->name(op.name() + "/mul");
            input_tensors_.emplace(&mul->input_a(), input_name);
            if (op.bottom(0) == op.top(0))
            {
                // inplace op, user op need this name
                output_tensors_.emplace(mul->name(), &mul->output());
            }
            else
                output_tensors_.emplace(op.top(0), &mul->output());
        }
        else
        {
            mul->name(op.name() + "/mul");
            auto beta = load_tensor<1>(op_data.blobs(1));
            std::vector<float> beta_vec_c(beta.begin(), beta.end());

            auto beta_const = graph_.emplace<constant>(dt_float32, shape_t { 1, input.shape()[1], 1, 1 }, beta_vec_c);
            beta_const->name(op.name() + "/beta_const");
            auto add = graph_.emplace<binary>(binary_add, dt_float32, mul->output().shape(), beta_const->output().shape(), value_range<float>::full());
            if (op.bottom(0) == op.top(0))
            {
                // inplace op, user op need this name
                add->name(op.top(0) + "/add");
            }
            else
                add->name(op.name() + "/add");
            add->input_a().connect(mul->output());
            add->input_b().connect(beta_const->output());
            input_tensors_.emplace(&mul->input_a(), input_name);
            if (op.bottom(0) == op.top(0))
            {
                // inplace op, user op need this name
                output_tensors_.emplace(add->name(), &add->output());
            }
            else
                output_tensors_.emplace(op.top(0), &add->output());
        }
    }
    else if (op.bottom_size() == 2)
    {
        // no emitter for broadcast, so two inputs is not supported now
        throw std::runtime_error("two inputs for scale is not supported yet");
    }
    else
        throw std::runtime_error("invalid bottom size for scale");
}
