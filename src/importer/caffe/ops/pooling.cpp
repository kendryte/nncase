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
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/reduce_window2d.h>

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
    auto ceil_mode = param.round_mode() ? false : true;

    auto pooling_method = param.pool();
    auto stride_h = param.global_pooling() ? 1 : (param.has_stride() ? param.stride() : (param.has_stride_h() ? param.stride_h() : 1));
    auto stride_w = param.global_pooling() ? 1 : (param.has_stride() ? param.stride() : (param.has_stride_w() ? param.stride_w() : 1));
    auto pad_h = param.global_pooling() ? 0 : (param.has_pad() ? param.pad() : (param.has_pad_h() ? param.pad_h() : 0));
    auto pad_w = param.global_pooling() ? 0 : (param.has_pad() ? param.pad() : (param.has_pad_w() ? param.pad_w() : 0));

    auto kernel_size_h = param.global_pooling() ? input.shape()[2] : (param.has_kernel_size() ? param.kernel_size() : param.kernel_h());
    auto kernel_size_w = param.global_pooling() ? input.shape()[3] : (param.has_kernel_size() ? param.kernel_size() : param.kernel_w());

    float init_value = 0.f;
    reduce_op_t reduce_type;

    if (pooling_method == 0)
    {
        reduce_type = reduce_max;
    }
    else if (pooling_method == 1)
    {
        reduce_type = reduce_mean;
    }
    else if (pooling_method == 2)
        throw std::runtime_error("STOCHASTIC pooling is not supported.");
    else
        throw std::runtime_error("wrong pooling type.");

    xt::svector<padding> paddings_rw;
    paddings_rw.push_back(padding { 0, 0 });
    paddings_rw.push_back(padding { 0, 0 });
    paddings_rw.push_back(padding { (int32_t)pad_h, (int32_t)pad_h });
    paddings_rw.push_back(padding { (int32_t)pad_w, (int32_t)pad_w });

    xt::svector<padding> paddings_ceil_align;
    auto ceil_align_h = 0;
    auto ceil_align_w = 0;
    if (ceil_mode)
    {
        if ((input.shape()[2] + 2 * pad_h - kernel_size_h + stride_h) % stride_h != 0)
            ceil_align_h = stride_h - (input.shape()[2] + 2 * pad_h - kernel_size_h + stride_h) % stride_h;
        if ((input.shape()[3] + 2 * pad_w - kernel_size_w + stride_w) % stride_w != 0)
            ceil_align_w = stride_w - (input.shape()[3] + 2 * pad_w - kernel_size_w + stride_w) % stride_w;
    }
    paddings_ceil_align.push_back(padding { 0, 0 });
    paddings_ceil_align.push_back(padding { 0, 0 });
    paddings_ceil_align.push_back(padding { 0, (int32_t)ceil_align_h });
    paddings_ceil_align.push_back(padding { 0, (int32_t)ceil_align_w });

    std::vector<int32_t> padding_h_w_after;
    padding_h_w_after.push_back((int32_t)ceil_align_h + (int32_t)pad_h);
    padding_h_w_after.push_back((int32_t)ceil_align_w + (int32_t)pad_w);

    auto p_rw = graph_.emplace<pad>(input.type(), input.shape(), paddings_rw, pad_constant, 0.f);
    p_rw->name(op.name() + "/pad");
    auto p_align = graph_.emplace<pad>(input.type(), p_rw->output().shape(), paddings_ceil_align, pad_constant, 0.f);
    p_align->name(op.name() + "/pad");
    auto rw = graph_.emplace<reduce_window2d>(reduce_type, p_align->output().shape(), init_value, kernel_size_h, kernel_size_w,
        padding { 0, 0 }, padding { 0, 0 }, stride_h, stride_w, 1, 1, value_range<float>::full(), ceil_mode, false, padding_h_w_after, true);
    rw->name(op.name() + "/reduce_window");

    input_tensors_.emplace(&p_rw->input(), input_name);
    p_align->input().connect(p_rw->output());
    rw->input().connect(p_align->output());
    output_tensors_.emplace(op.top(0), &rw->output());
}
