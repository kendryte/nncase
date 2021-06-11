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

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace caffe;

DEFINE_CAFFE_LOWER(LSTM)
{
    auto &input_a = *output_tensors_.at(op.bottom(0));
    auto &input_b = *output_tensors_.at(op.bottom(1));
    [[maybe_unused]]auto &param = op.recurrent_param();
    [[maybe_unused]]auto n_output = param.num_output();

    std::cout<<"bsize: "<<op.blobs().size()<<std::endl;
    // std::cout<<"bs0: "<<op.blobs(0).shape().dim_size()<<std::endl;
    // std::cout<<"bs1: "<<op.blobs(1).shape().dim_size()<<std::endl;
    // std::cout<<"bs2: "<<op.blobs(2).shape().dim_size()<<std::endl;

    // std::cout<<"bs00: "<<op.blobs(0).shape().dim(0)<<std::endl;
    // std::cout<<"bs01: "<<op.blobs(0).shape().dim(1)<<std::endl;
    // std::cout<<"bs10: "<<op.blobs(1).shape().dim(0)<<std::endl;
    // std::cout<<"bs20: "<<op.blobs(2).shape().dim(0)<<std::endl;
    // std::cout<<"bs21: "<<op.blobs(2).shape().dim(1)<<std::endl;
    // [[maybe_unused]]auto t1 = load_tensor<2>(op.blobs(0));
    // [[maybe_unused]]auto t2 = load_tensor<1>(op.blobs(1));
    // [[maybe_unused]]auto t3 = load_tensor<2>(op.blobs(2));

    std::cout<<"test1"<<std::endl;
    for (size_t i = 0;i<input_a.shape().size();i++)
        std::cout<<input_a.shape()[i]<<std::endl;
    std::cout<<"test2"<<std::endl;
    for (size_t i = 0;i<input_b.shape().size();i++)
        std::cout<<input_b.shape()[i]<<std::endl;

    auto node = graph_.emplace<lstm>(input_a.shape(), input_b.shape(), n_output);
    node->name(op.name() + "/lstm");

    input_tensors_.emplace(&node->input_a(), op.bottom(0));
    input_tensors_.emplace(&node->input_b(), op.bottom(1));
    output_tensors_.emplace(op.top(0), &node->output());
}