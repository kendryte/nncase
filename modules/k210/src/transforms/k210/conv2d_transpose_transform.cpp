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
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/conv2d_transpose.h>
#include <nncase/ir/ops/k210/fake_kpu_conv2d.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/reduce_window2d.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/quantizer.h>
#include <nncase/ir/visitor.h>
#include <nncase/runtime/k210/runtime_op_utility.h>
#include <nncase/targets/target.h>
#include <nncase/transforms/k210/conv2d_transpose_transform.h>
#include <nncase/transforms/k210/fake_kpu_conv2d.h>
#include <nncase/transforms/k210/kpu_utils.h>
#include <xtensor/xadapt.hpp>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime::k210;
using namespace nncase::ir::transforms;
using namespace nncase::ir::transforms::k210;

#define GET_PRE_PAD(conv)                                                      \
    auto filter_type = get_filter_type(conv.filter_h());                       \
    auto kpu_pad = get_kpu_padding(filter_type);                               \
    padding pad_h{conv.padding_h().before - kpu_pad,                           \
                  conv.padding_h().after - kpu_pad};                           \
    padding pad_w{conv.padding_w().before - kpu_pad,                           \
                  conv.padding_w().after - kpu_pad};                           \
                                                                               \
    [[maybe_unused]] auto pre_pad_h = get_padding<true>(pad_h);                \
    [[maybe_unused]] auto pre_pad_w = get_padding<true>(pad_w);

bool conv2d_transpose_transform::on_try_match(node &node,
                                              transform_context &context) {
    conv2d_transpose *conv;
    constant *weights;
    constant *bias;
    if ((conv = node_cast<conv2d_transpose>(node)) &&
        (weights = try_get_direct_parent<constant>(*conv, 1)) &&
        (bias = try_get_direct_parent<constant>(*conv, 2))) {
        if ((conv->groups() == 1 || conv->groups() == conv->input_channels()) &&
            conv->dilation_h() == 1 && conv->dilation_w() == 1 &&
            is_supported_filter(conv->filter_h(), conv->filter_w()) &&
            is_supported_in_shape(conv->input().shape()) // TODO
            && is_supported_out_shape(conv->output().shape()) &&
            !is_bad_shape(conv->input().shape(), conv->output().shape())) {
            GET_PRE_PAD((*conv));
            auto new_in_shape = conv->input().shape();
            new_in_shape[2] += pre_pad_h.sum();
            new_in_shape[3] += pre_pad_w.sum();

            if (is_supported_in_shape(new_in_shape)) {
                context.inputs.emplace_back(&conv->input());
                context.inputs.emplace_back(&conv->bias());
                context.outputs.emplace_back(&conv->output());

                context.matched_nodes.emplace_back(conv);
                context.matched_nodes.emplace_back(weights);
                return true;
            }
        }
    }

    return false;
}

// conv2d_transpose -> pad + conv2d
void conv2d_transpose_transform::process(transform_context &context) {
    auto &output = *context.inputs[0]->connection();
    auto &bias = *context.inputs[1]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &ct = static_cast<conv2d_transpose &>(*context.matched_nodes[0]);
    auto &old_filter = static_cast<constant &>(*context.matched_nodes[1]);

    // pad
    xt::svector<padding> paddings = {padding::zero(), padding::zero(),
                                     ct.padding_h(), ct.padding_w()};
    paddings[2].after += ct.output_padding_h();
    paddings[2].interior = ct.stride_h() - 1;
    paddings[3].after += ct.output_padding_w();
    paddings[3].interior = ct.stride_w() - 1;

    auto pre_pad = context.graph.emplace<pad>(dt_float32, output.shape(),
                                              paddings, pad_constant, 0.f);
    pre_pad->name(ct.name() + "/Pad");

    // reverse weight
    auto buf = reinterpret_cast<const float *>(old_filter.data().data());
    auto filter_shape = ct.weights().shape();
    auto filter_nc = filter_shape[0] * filter_shape[1];
    auto filter_hw = filter_shape[2] * filter_shape[3];
    std::vector<float> v(buf, buf + filter_nc * filter_hw);
    for (size_t i = 0; i < filter_nc; i++) {
        auto begin = v.begin() + i * filter_hw;
        std::reverse(begin, begin + filter_hw);
    }
    auto new_filter =
        context.graph.emplace<constant>(dt_float32, ct.weights().shape(), v);
    new_filter->name(ct.name() + "/Weight");

    // conv2d
    auto conv = context.graph.emplace<conv2d>(
        pre_pad->output().shape(), new_filter->output().shape(), ct.groups(),
        padding::zero(), padding::zero(), 1, 1, ct.dilation_h(),
        ct.dilation_w(), value_range<float>::full());
    conv->name(ct.name() + "/Conv2d");

    pre_pad->input().connect(output);
    conv->input().connect(pre_pad->output());
    conv->weights().connect(new_filter->output());
    conv->bias().connect(bias);
    for (auto &in : dup(inputs))
        in->connect(conv->output());
}