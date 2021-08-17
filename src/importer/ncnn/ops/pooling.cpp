// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "../ncnn_importer.h"
#include <cassert>
#include <nncase/ir/graph.h>
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/reduce_window2d.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace ncnn;

void nncase::importer::ncnn_importer::convert_op_Pooling(const Layer &layer, const ParamDict &pd, const ModelBin& /*mb*/)
{
    const int pooling_type = pd.get(0, 0);
    const int kernel_w = pd.get(1, 0);
    const int kernel_h = pd.get(11, kernel_w);
    const int stride_w = pd.get(2, 1);
    const int stride_h = pd.get(12, stride_w);
    const int pad_left = pd.get(3, 0);
    const int pad_right = pd.get(14, pad_left);
    const int pad_top = pd.get(13, pad_left);
    const int pad_bottom = pd.get(15, pad_top);
    const int global_pooling = pd.get(4, 0);
    const int pad_mode = pd.get(5, 0);
    const int avgpool_count_include_pad = pd.get(6, 0);
    const int adaptive_pooling = pd.get(7, 0);
    const int out_w = pd.get(8, 0);
    const int out_h = pd.get(18, out_w);

    const auto &op_name = layer.name;

    auto in_shape = layer.bottom_shapes[0];

    reduce_op_t reduce_type;
    float init_value;
    if (pooling_type == 0)
    {
        reduce_type = reduce_max;
        init_value = -FLT_MAX;
    }
    if (pooling_type == 1)
    {
        reduce_type = reduce_mean;
        init_value = 0.f;
    }

    value_range<float> fused_activation = value_range<float>::full();

    bool ceil_mode = false;

    if (global_pooling)
    {
        axis_t axis = {2, 3};
        bool keep_dims = false;
        ir::reduce* reduce_op = graph_.emplace<reduce>(reduce_type, in_shape, axis, init_value, keep_dims);
        reduce_op->name(op_name + ".reduce(Pooling)");

        input_tensors_.emplace(&reduce_op->input(), layer.bottoms[0]);
        output_tensors_.emplace(layer.tops[0], &reduce_op->output());
    }
    if (adaptive_pooling)
    {
        const int w = in_shape[2];
        const int h = in_shape[1];
        const int kernel_extent_h = h - out_h + 1;
        const int kernel_extent_w = w - out_w + 1;

        ir::reduce_window2d* reduce_window2d_op = graph_.emplace<reduce_window2d>(reduce_type, in_shape, init_value, kernel_extent_h, kernel_extent_w, padding{0, 0}, padding{0, 0}, 1, 1, 1, 1, fused_activation, ceil_mode, avgpool_count_include_pad);
        reduce_window2d_op->name(op_name + ".reduce_window2d(Pooling)");

        input_tensors_.emplace(&reduce_window2d_op->input(), layer.bottoms[0]);
        output_tensors_.emplace(layer.tops[0], &reduce_window2d_op->output());
    }

    padding padding_h;
    padding padding_w;
    {
        if (pad_mode == 0) // full padding
        {
            const int w = in_shape[2];
            const int h = in_shape[1];
            const int wtail = (w + pad_left + pad_right - kernel_w) % stride_w;
            const int htail = (h + pad_top + pad_bottom - kernel_h) % stride_h;

            int wtailpad = 0;
            int htailpad = 0;

            if (wtail != 0)
                wtailpad = stride_w - wtail;
            if (htail != 0)
                htailpad = stride_h - htail;

            padding_h = {pad_top, pad_bottom + htailpad};
            padding_w = {pad_left, pad_right + wtailpad};
        }
        if (pad_mode == 1) // valid padding
        {
            padding_h = {pad_top, pad_bottom};
            padding_w = {pad_left, pad_right};
        }
        if (pad_mode == 2) // tensorflow padding=SAME or onnx padding=SAME_UPPER
        {
            const int w = in_shape[2];
            const int h = in_shape[1];
            const int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
            const int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;

            padding_h = {hpad / 2, hpad - hpad / 2};
            padding_w = {wpad / 2, wpad - wpad / 2};
        }
        if (pad_mode == 3) // onnx padding=SAME_LOWER
        {
            const int w = in_shape[2];
            const int h = in_shape[1];
            const int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
            const int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;

            padding_h = {hpad - hpad / 2, hpad / 2};
            padding_w = {wpad - wpad / 2, wpad / 2};
        }
    }

    ir::reduce_window2d* reduce_window2d_op = graph_.emplace<reduce_window2d>(reduce_type, in_shape, init_value, kernel_h, kernel_w, padding_h, padding_w, stride_h, stride_w, 1, 1, fused_activation, ceil_mode, avgpool_count_include_pad);
    reduce_window2d_op->name(op_name + ".reduce_window2d(Pooling)");

    input_tensors_.emplace(&reduce_window2d_op->input(), layer.bottoms[0]);
    output_tensors_.emplace(layer.tops[0], &reduce_window2d_op->output());
}
