/* Copyright 2019 Canaan Inc.
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
#include <ir/ops/conv2d.h>
#include <ir/ops/k210/fake_kpu_conv2d.h>
#include <ir/ops/pad.h>
#include <ir/ops/strided_slice.h>
#include <ir/visitor.h>
#include <runtime/k210/k210_runtime_op_utility.h>
#include <transforms/k210/fake_kpu_conv2d.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime::k210;
using namespace nncase::transforms;
using namespace nncase::transforms::k210;

namespace
{
bool is_supported_in_shape(const shape_t &in_shape)
{
    return in_shape[0] == 1 && in_shape[1] <= 1024 && in_shape[2] >= 4 && in_shape[2] <= 256 && in_shape[3] >= 4 && in_shape[3] <= 512;
}

bool is_supported_out_shape(const shape_t &in_shape)
{
    return in_shape[0] == 1 && in_shape[1] <= 1024;
}

bool is_supported_filter(int32_t filter_h, int32_t filter_w)
{
    return (filter_h == filter_w) && (filter_h == 3 || filter_h == 1);
}

template <bool Pre>
padding get_padding(const padding &padding)
{
    if (Pre)
        return { padding.before > 0 ? padding.before : 0, padding.after > 0 ? padding.after : 0 };
    else
        return { padding.before < 0 ? padding.before : 0, padding.after < 0 ? padding.after : 0 };
}

kpu_filter_type_t get_filter_type(int32_t filter)
{
    return filter == 1 ? kpu_filter_1x1 : kpu_filter_3x3;
}

xt::svector<piecewise_linear_segment> clamp_to_piecewise(value_range<float> clamp)
{
    xt::svector<piecewise_linear_segment> segs;
    if (clamp.min != std::numeric_limits<float>::min())
        segs.push_back({ std::numeric_limits<float>::min(), 0.f, clamp.min });
    segs.push_back({ clamp.min, 1.f, 0.f });
    if (clamp.max != std::numeric_limits<float>::max())
        segs.push_back({ clamp.max, 0.f, clamp.max });
    return segs;
}
}

bool fake_kpu_conv2d_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_conv2d)
    {
        auto &conv = static_cast<conv2d &>(node);
        if ((conv.groups() == 1 || conv.groups() == conv.input_channels())
            && conv.dilation_h() == 1 && conv.dilation_w() == 1
            && is_supported_filter(conv.filter_h(), conv.filter_w())
            && is_supported_in_shape(conv.input().shape())
            && is_supported_out_shape(conv.output().shape()))
        {
            context.inputs.emplace_back(&conv.input());
            context.outputs.emplace_back(&conv.output());

            context.matched_nodes.emplace_back(&conv);
            return true;
        }
    }

    return false;
}

void fake_kpu_conv2d_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_conv = static_cast<conv2d &>(*context.matched_nodes[0]);

    auto is_depthwise = old_conv.groups() == old_conv.input_channels();
    auto filter_type = get_filter_type(old_conv.filter_h());
    auto kpu_pad = get_kpu_padding(filter_type);
    padding pad_h { old_conv.padding_h().before - kpu_pad, old_conv.padding_h().after - kpu_pad };
    padding pad_w { old_conv.padding_w().before - kpu_pad, old_conv.padding_w().after - kpu_pad };
    xt::svector<padding> pre_paddings {
        padding::zero(),
        padding::zero(),
        get_padding<true>(pad_h),
        get_padding<true>(pad_w)
    };

    auto pre_pad = context.graph.emplace<pad>(dt_float32, output.shape(), pre_paddings, 0.f);
    auto conv = context.graph.emplace<fake_kpu_conv2d>(pre_pad->output().shape(), is_depthwise, filter_type, kpu_pool_bypass,
        old_conv.weights(), old_conv.bias(), clamp_to_piecewise(old_conv.fused_activation()));

    xt::svector<padding> sur_paddings {
        padding::zero(),
        padding::zero(),
        get_padding<false>(pad_h),
        get_padding<false>(pad_w)
    };
    axis_t strides { 1, 1, old_conv.stride_h(), old_conv.stride_w() };
    auto sur_pad = context.graph.emplace<pad>(dt_float32, conv->output().shape(), sur_paddings, 0.f);
    auto slice = context.graph.emplace<strided_slice>(dt_float32, sur_pad->output().shape(), axis_t { 0, 0, 0, 0 }, axis_t { 0, 0, 0, 0 }, strides, 15, 15, 0, 0, 0);
    conv->input().connect(pre_pad->output());
    sur_pad->input().connect(conv->output());
    slice->input().connect(sur_pad->output());

    pre_pad->input().connect(output);

    for (auto &in : dup(inputs))
        in->connect(slice->output());
}

bool fuse_fake_kpu_conv2d_strided_slice_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_k210_fake_kpu_conv2d)
    {
        auto &conv = static_cast<fake_kpu_conv2d &>(node);
        if (!conv.is_depthwise())
        {
            if (auto slice = try_get_direct_child<strided_slice>(conv))
            {
                if (slice->strides() == axis_t { 1, 1, 2, 2 }
                    && is_supported_in_shape(slice->output().shape()))
                {
                    context.inputs.emplace_back(&conv.input());
                    context.outputs.emplace_back(&slice->output());

                    context.matched_nodes.emplace_back(&conv);
                    context.matched_nodes.emplace_back(slice);
                    return true;
                }
            }
        }
    }

    return false;
}

void fuse_fake_kpu_conv2d_strided_slice_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_conv = static_cast<fake_kpu_conv2d &>(*context.matched_nodes[0]);
    auto &old_slice = static_cast<strided_slice &>(*context.matched_nodes[1]);

    padding pad_h { old_slice.begin()[2] % 2, 0 };
    padding pad_w { 0, 0 };
    // pad to even
    if ((old_conv.input().shape()[2] + pad_h.before) % 2 == 1)
        pad_h.after += 1;
    if (old_conv.input().shape()[3] % 2 == 1)
        pad_w.after += 1;
    auto pool_type = old_slice.begin()[3] % 2 == 0 ? kpu_pool_left_top_2_s2 : kpu_pool_right_top_2_s2;

    auto p = context.graph.emplace<pad>(dt_float32, output.shape(), xt::svector<padding> { padding::zero(), padding::zero(), pad_h, pad_w }, 0.f);
    auto conv = context.graph.emplace<fake_kpu_conv2d>(p->output().shape(), old_conv.is_depthwise(), old_conv.filter_type(), pool_type,
        old_conv.weights(), old_conv.bias(), old_conv.fused_activation());

    padding crop_h { -(old_slice.begin()[2] / 2 + pad_h.before), (old_slice.end()[2] - old_slice.input().shape()[2]) / 2 };
    padding crop_w { -(old_slice.begin()[3] / 2), (old_slice.end()[3] - old_slice.input().shape()[3]) / 2 };

    auto crop = context.graph.emplace<pad>(dt_float32, conv->output().shape(), xt::svector<padding> { padding::zero(), padding::zero(), crop_h, crop_w }, 0.f);
    conv->input().connect(p->output());
    crop->input().connect(conv->output());

    p->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(crop->output());
}