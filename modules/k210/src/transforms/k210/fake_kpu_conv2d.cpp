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
#include <nncase/ir/ops/k210/fake_kpu_conv2d.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/reduce_window2d.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/quantizer.h>
#include <nncase/ir/visitor.h>
#include <nncase/runtime/k210/runtime_op_utility.h>
#include <nncase/targets/target.h>
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

bool fake_kpu_conv2d_transform::on_try_match(node &node,
                                             transform_context &context) {
    conv2d *conv;
    constant *weights;
    constant *bias;
    if ((conv = node_cast<conv2d>(node)) &&
        (weights = try_get_direct_parent<constant>(*conv, 1)) &&
        (bias = try_get_direct_parent<constant>(*conv, 2))) {
        //{
        //    auto total_range =
        //    quantizer::fixup_range(quantizer::get_range(weights.begin(),
        //    weights.end())); if (total_range.max - total_range.min >
        //    context.target.options().weights_quantize_threshold)
        //        return false;
        //}

        if ((conv->groups() == 1 || conv->groups() == conv->input_channels()) &&
            conv->dilation_h() == 1 && conv->dilation_w() == 1 &&
            is_supported_filter(conv->filter_h(), conv->filter_w()) &&
            is_supported_in_shape(conv->input().shape()) &&
            is_supported_out_shape(conv->output().shape()) &&
            !is_bad_shape(conv->input().shape(), conv->output().shape())) {
            GET_PRE_PAD((*conv));
            auto new_in_shape = conv->input().shape();
            new_in_shape[2] += pre_pad_h.sum();
            new_in_shape[3] += pre_pad_w.sum();

            if (is_supported_in_shape(new_in_shape)) {
                context.inputs.emplace_back(&conv->input());
                context.inputs.emplace_back(&conv->weights());
                context.inputs.emplace_back(&conv->bias());
                context.outputs.emplace_back(&conv->output());

                context.matched_nodes.emplace_back(conv);
                return true;
            }
        }
    }

    return false;
}

void fake_kpu_conv2d_transform::process(transform_context &context) {
    auto &output = *context.inputs[0]->connection();
    auto &weights = *context.inputs[1]->connection();
    auto &bias = *context.inputs[2]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_conv = static_cast<conv2d &>(*context.matched_nodes[0]);

    auto is_depthwise =
        old_conv.input_channels() == old_conv.output_channels() &&
        old_conv.output_channels() == old_conv.groups();
    GET_PRE_PAD(old_conv);
    xt::svector<padding> pre_paddings{padding::zero(), padding::zero(),
                                      get_padding<true>(pad_h),
                                      get_padding<true>(pad_w)};

    auto pre_pad = context.graph.emplace<pad>(dt_float32, output.shape(),
                                              pre_paddings, pad_constant, 0.f);
    auto conv = context.graph.emplace<fake_kpu_conv2d>(
        pre_pad->output().shape(), is_depthwise, weights.shape(), filter_type,
        kpu_pool_bypass, old_conv.fused_activation());
    conv->name(old_conv.name());

    xt::svector<padding> sur_paddings{padding::zero(), padding::zero(),
                                      get_padding<false>(pad_h),
                                      get_padding<false>(pad_w)};
    axis_t strides{1, 1, old_conv.stride_h(), old_conv.stride_w()};
    auto sur_pad = context.graph.emplace<pad>(
        dt_float32, conv->output().shape(), sur_paddings, pad_constant, 0.f);
    auto slc = context.graph.emplace<slice>(
        dt_float32, sur_pad->output().shape(), axis_t{0, 0, 0, 0},
        axis_t{0, 0, 0, 0}, strides, 15, 15, 0, 0);
    conv->input().connect(pre_pad->output());
    conv->weights().connect(weights);
    conv->bias().connect(bias);
    sur_pad->input().connect(conv->output());
    slc->input().connect(sur_pad->output());

    pre_pad->input().connect(output);

    for (auto &in : dup(inputs))
        in->connect(slc->output());
}

#undef GET_PRE_PAD
#define GET_PRE_PAD(conv, slice)                                               \
    padding pad_h{slice.begin()[2] % 2, 0};                                    \
    padding pad_w{0, 0};                                                       \
    /* pad to even */                                                          \
    if ((slice.input().shape()[2] + pad_h.before) % 2 == 1)                    \
        pad_h.after += 1;                                                      \
    if (conv.input().shape()[3] % 2 == 1)                                      \
        pad_w.after += 1;

bool fuse_fake_kpu_conv2d_strided_slice_transform::on_try_match(
    node &node, transform_context &context) {
    if (node.runtime_opcode() == op_k210_fake_kpu_conv2d) {
        auto &conv = static_cast<fake_kpu_conv2d &>(node);
        if (!conv.is_depthwise()) {
            if (auto slice_p = try_get_direct_child<slice>(conv)) {
                auto &slice = *slice_p;
                if (slice.strides() == axis_t{1, 1, 2, 2} &&
                    is_supported_in_shape(slice.output().shape())) {
                    GET_PRE_PAD(conv, slice);
                    auto new_in_shape = conv.input().shape();
                    new_in_shape[2] += pad_h.sum();
                    new_in_shape[3] += pad_w.sum();

                    if (is_supported_in_shape(new_in_shape)) {
                        context.inputs.emplace_back(&conv.input());
                        context.inputs.emplace_back(&conv.weights());
                        context.inputs.emplace_back(&conv.bias());
                        context.outputs.emplace_back(&slice.output());

                        context.matched_nodes.emplace_back(&conv);
                        context.matched_nodes.emplace_back(&slice);
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

void fuse_fake_kpu_conv2d_strided_slice_transform::process(
    transform_context &context) {
    auto &output = *context.inputs[0]->connection();
    auto &weights = *context.inputs[1]->connection();
    auto &bias = *context.inputs[2]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_conv = static_cast<fake_kpu_conv2d &>(*context.matched_nodes[0]);
    auto &old_slice = static_cast<slice &>(*context.matched_nodes[1]);

    GET_PRE_PAD(old_conv, old_slice);
    auto pool_type = old_slice.begin()[3] % 2 == 0 ? kpu_pool_left_top_2_s2
                                                   : kpu_pool_right_top_2_s2;

    auto p = context.graph.emplace<pad>(
        dt_float32, output.shape(),
        xt::svector<padding>{padding::zero(), padding::zero(), pad_h, pad_w},
        pad_constant, 0.f);
    auto conv = context.graph.emplace<fake_kpu_conv2d>(
        p->output().shape(), old_conv.is_depthwise(), weights.shape(),
        old_conv.filter_type(), pool_type, old_conv.fused_activation());
    conv->name(old_conv.name());

    padding crop_h{-(old_slice.begin()[2] / 2 + pad_h.before)};
    padding crop_w{-(old_slice.begin()[3] / 2)};
    crop_h.after = old_slice.output().shape()[2] - conv->output().shape()[2] -
                   crop_h.before;
    crop_w.after = old_slice.output().shape()[3] - conv->output().shape()[3] -
                   crop_w.before;

    auto crop = context.graph.emplace<pad>(
        dt_float32, conv->output().shape(),
        xt::svector<padding>{padding::zero(), padding::zero(), crop_h, crop_w},
        pad_constant, 0.f);
    conv->input().connect(p->output());
    conv->weights().connect(weights);
    conv->bias().connect(bias);
    crop->input().connect(conv->output());

    p->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(crop->output());
}
