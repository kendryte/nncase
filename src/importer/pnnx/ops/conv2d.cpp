// Tencent is pleased to support the open source community by making pnnx available.
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

#include "../pnnx_importer.h"
#include "nncase/importer/util.h"
#include "nncase/ir/ir_types.h"
#include <cassert>
#include <nncase/ir/graph.h>
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/placeholders.h>
#include <stdexcept>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace pnnx;

void nncase::importer::pnnx_importer::convert_op_nn_Conv2d(const Operator &op)
{
    const auto &op_name = op.name;

    auto in_shape = op.inputs[0]->get_shape();
    auto weight_shape = op.attrs.at("op_0.weight").get_shape();

    const int dilation_w = op.params.at("dilation").ai[1];
    const int dilation_h = op.params.at("dilation").ai[0];
    const int stride_w = op.params.at("stride").ai[1];
    const int stride_h = op.params.at("stride").ai[0];
    const int group = op.params.at("group").i;
    std::string padding_mode = op.params.at("padding_mode").s;

    padding padding_w;
    padding padding_h;
    if (op.params.at("padding").type == 4)
    {
        if (op.params.at("padding").s == "same")
        {
            padding_w = get_windowed_padding(in_shape[3], weight_shape[3], stride_w, dilation_w, true);
            padding_h = get_windowed_padding(in_shape[2], weight_shape[2], stride_h, dilation_h, true);
        }
        else // if (op.params.at("padding").s == "valid")
        {
            padding_w = { 0, 0 };
            padding_h = { 0, 0 };
        }
    }
    else
    {
        padding_w = { op.params.at("padding").ai[1], op.params.at("padding").ai[1] };
        padding_h = { op.params.at("padding").ai[0], op.params.at("padding").ai[0] };
    }

    ir::pad *pad_op = 0;
    if (padding_mode == "reflect" || padding_mode == "replicate")
    {
        xt::svector<padding> paddings = { { 0, 0 }, { 0, 0 }, padding_h, padding_w };
        pad_mode_t pad_mode = padding_mode == "reflect" ? pad_reflect : pad_edge;

        pad_op = graph_.emplace<pad>(dt_float32, in_shape, paddings, pad_mode, 0.f);
        pad_op->name(op_name + ".pad(Convolution)");

        padding_w = { 0, 0 };
        padding_h = { 0, 0 };
    }

    ir::conv2d *conv_op = graph_.emplace<conv2d>(in_shape, weight_shape, group, padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w, value_range<float>::full());
    conv_op->name(op_name + ".conv2d(Conv2d)");

    if (pad_op)
    {
        conv_op->input().connect(pad_op->output());
    }

    auto weight_data = op.attrs.at("op_0.weight").get_data();

    auto weight_node = graph_.emplace<constant>(dt_float32, weight_shape, weight_data);
    conv_op->weights().connect(weight_node->output());

    if (op.params.at("bias").b)
    {
        auto bias_shape = op.attrs.at("op_0.bias").get_shape();
        auto bias_data = op.attrs.at("op_0.bias").get_data();

        auto bias_node = graph_.emplace<constant>(dt_float32, bias_shape, bias_data);
        conv_op->bias().connect(bias_node->output());
    }

    if (pad_op)
    {
        input_tensors_.emplace(&pad_op->input(), op.inputs[0]->name);
    }
    else
    {
        input_tensors_.emplace(&conv_op->input(), op.inputs[0]->name);
    }

    output_tensors_.emplace(op.outputs[0]->name, &conv_op->output());
}
