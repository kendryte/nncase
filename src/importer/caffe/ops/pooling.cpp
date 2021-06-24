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
#include <nncase/ir/ops/reduce_window2d.h>
#include <nncase/ir/ops/constant.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace caffe;

DEFINE_CAFFE_LOWER(Pooling)
{
    // check if there are bn/scale/relu above
    std::string input_name = get_real_input_names(op)[0];

    auto &input = *output_tensors_.at(input_name);
    auto &param = op.pooling_param();

    auto pooling_method = param.pool();
    auto stride_h = param.has_stride() ? param.stride() : (param.has_stride_h() ? param.stride_h() : 1);
    auto stride_w = param.has_stride() ? param.stride() : (param.has_stride_w() ? param.stride_w() : 1);
    auto pad_h = param.has_pad() ? param.pad() : (param.has_pad_h() ? param.pad_h() : 0);
    auto pad_w = param.has_pad() ? param.pad() : (param.has_pad_w() ? param.pad_w() : 0);
    if ((!param.has_kernel_size() && !param.has_kernel_h()) || (!param.has_kernel_size() && !param.has_kernel_w()))
        throw std::runtime_error("import pooling error.");
    auto kernel_size_h = param.has_kernel_size() ? param.kernel_size() : param.kernel_h();
    auto kernel_size_w = param.has_kernel_size() ? param.kernel_size() : param.kernel_w();

    float init_value = 0.f;
    reduce_op_t reduce_type;

    if (pooling_method == 0)
    {
        reduce_type = reduce_max;
        init_value = std::numeric_limits<float>::lowest();
    }
    else if (pooling_method == 1)
    {
        reduce_type = reduce_mean;
    }
    else if (pooling_method == 2)
        throw std::runtime_error("STOCHASTIC pooling is not supported yet.");
    else
        throw std::runtime_error("wrong pooling type.");

    auto node = graph_.emplace<reduce_window2d>(reduce_type, input.shape(), init_value, kernel_size_h, kernel_size_w,
                                                padding { (int32_t)pad_h, (int32_t)pad_h }, padding { (int32_t)pad_w, (int32_t)pad_w }, stride_h, stride_w, 1, 1, value_range<float>::full());
    node->name(op.name() + "/reduce_window");

    input_tensors_.emplace(&node->input(), input_name);
    output_tensors_.emplace(op.top(0), &node->output());
}
