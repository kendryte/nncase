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
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/fold_1x1_conv2d_depthwise_conv2d.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool fold_1x1_conv2d_depthwise_conv2d_transform::on_try_match(node &node, transform_context &context)
{
    conv2d *conv1 = nullptr;
    conv2d *conv2 = nullptr;
    if ((conv1 = node_cast<conv2d>(node))
        && (conv2 = try_get_direct_child<conv2d>(*conv1))
        && (!conv1->is_depthwise() && conv2->is_depthwise())
        && (conv1->filter_h() == 1 && conv1->filter_w() == 1)
        && (conv1->fused_activation() == value_range<float>::full())
        && conv1->outputs().size() == 1 && conv1->groups() == 1)
    {
        context.inputs.emplace_back(&conv1->input());
        context.outputs.emplace_back(&conv2->output());

        context.matched_nodes.emplace_back(conv1);
        context.matched_nodes.emplace_back(conv2);
        context.matched_nodes.emplace_back(&conv1->weights().connection()->owner());
        context.matched_nodes.emplace_back(&conv1->bias().connection()->owner());
        context.matched_nodes.emplace_back(&conv2->weights().connection()->owner());
        context.matched_nodes.emplace_back(&conv2->bias().connection()->owner());

        return true;
    }

    return false;
}

void fold_1x1_conv2d_depthwise_conv2d_transform::process(transform_context &context)
{
    auto old_conv1 = node_cast<conv2d>(*context.matched_nodes[0]);
    auto old_conv2 = node_cast<conv2d>(*context.matched_nodes[1]);
    auto old_w1 = node_cast<constant>(*context.matched_nodes[2]);
    auto old_b1 = node_cast<constant>(*context.matched_nodes[3]);
    auto old_w2 = node_cast<constant>(*context.matched_nodes[4]);
    auto old_b2 = node_cast<constant>(*context.matched_nodes[5]);

    const float *w1 = reinterpret_cast<const float *>(old_w1->data().data());
    const float *b1 = reinterpret_cast<const float *>(old_b1->data().data());
    const float *w2 = reinterpret_cast<const float *>(old_w2->data().data());
    const float *b2 = reinterpret_cast<const float *>(old_b2->data().data());

    // new weights
    std::vector<float> weights(old_conv1->output_channels() * old_conv1->input_channels() * old_conv2->filter_h() * old_conv2->filter_w(), 0.f);
    for (size_t i = 0; i < weights.size(); i++)
    {
        auto m = i / (old_conv1->input_channels() * old_conv2->filter_h() * old_conv2->filter_w());
        auto remainder = i % (old_conv1->input_channels() * old_conv2->filter_h() * old_conv2->filter_w());
        auto c = remainder / (old_conv2->filter_h() * old_conv2->filter_w());
        remainder = remainder % (old_conv2->filter_h() * old_conv2->filter_w());
        auto r = remainder / old_conv2->filter_w();
        auto s = remainder % old_conv2->filter_w();
        weights[i] = w1[m * old_conv1->input_channels() + c] * w2[m * old_conv2->filter_h() * old_conv2->filter_w() + r * old_conv2->filter_w() + s];
    }
    auto new_weights = context.graph.emplace<constant>(dt_float32, shape_t { static_cast<unsigned long>(old_conv1->output_channels()), static_cast<unsigned long>(old_conv1->input_channels()), static_cast<unsigned long>(old_conv2->filter_h()), static_cast<unsigned long>(old_conv2->filter_w()) }, std::span(weights));
    new_weights->name(old_w1->name() + "_fuse_" + old_w2->name());

    // new bias
    std::vector<float> bias(old_conv1->output_channels(), 0.f);
    for (size_t m = 0; m < bias.size(); m++)
    {
        for (size_t r = 0; r < old_conv2->filter_h(); r++)
            for (size_t s = 0; s < old_conv2->filter_w(); s++)
                bias[m] += b1[m] * w2[m * old_conv2->filter_h() * old_conv2->filter_w() + r * old_conv2->filter_w() + s];
        bias[m] += b2[m];
    }
    auto new_bias = context.graph.emplace<constant>(dt_float32, shape_t { static_cast<unsigned long>(old_conv1->output_channels()) }, std::span(bias));
    new_bias->name(old_b1->name() + "_fuse_" + old_b2->name());

    // create new conv2d
    auto new_conv = context.graph.emplace<conv2d>(old_conv1->input().shape(), new_weights->output().shape(), old_conv1->groups(), old_conv2->padding_h(), old_conv2->padding_w(), old_conv2->stride_h(), old_conv2->stride_w(), old_conv2->dilation_h(), old_conv2->dilation_w(), old_conv2->fused_activation());

    new_conv->input().connect(*context.inputs[0]->connection());
    new_conv->weights().connect(new_weights->output());
    new_conv->bias().connect(new_bias->output());
    new_conv->name(old_conv1->name() + "_fuse_" + old_conv2->name());

    auto inputs = context.outputs[0]->connections();
    for (auto &in : dup(inputs))
        in->connect(new_conv->output());
}

bool fold_depthwise_conv2d_1x1_conv2d_transform::on_try_match(node &node, transform_context &context)
{
    conv2d *conv1 = nullptr;
    conv2d *conv2 = nullptr;
    if ((conv1 = node_cast<conv2d>(node))
        && (conv2 = try_get_direct_child<conv2d>(*conv1))
        && (conv1->is_depthwise() && !conv2->is_depthwise())
        && (conv2->filter_h() == 1 && conv2->filter_w() == 1)
        && (conv1->fused_activation() == value_range<float>::full())
        && conv1->outputs().size() == 1 && conv2->groups() == 1)
    {
        context.inputs.emplace_back(&conv1->input());
        context.outputs.emplace_back(&conv2->output());

        context.matched_nodes.emplace_back(conv1);
        context.matched_nodes.emplace_back(conv2);
        context.matched_nodes.emplace_back(&conv1->weights().connection()->owner());
        context.matched_nodes.emplace_back(&conv1->bias().connection()->owner());
        context.matched_nodes.emplace_back(&conv2->weights().connection()->owner());
        context.matched_nodes.emplace_back(&conv2->bias().connection()->owner());

        return true;
    }

    return false;
}

void fold_depthwise_conv2d_1x1_conv2d_transform::process([[maybe_unused]] transform_context &context)
{
    auto old_conv1 = node_cast<conv2d>(*context.matched_nodes[0]);
    auto old_conv2 = node_cast<conv2d>(*context.matched_nodes[1]);
    auto old_w1 = node_cast<constant>(*context.matched_nodes[2]);
    auto old_b1 = node_cast<constant>(*context.matched_nodes[3]);
    auto old_w2 = node_cast<constant>(*context.matched_nodes[4]);
    auto old_b2 = node_cast<constant>(*context.matched_nodes[5]);

    const float *w1 = reinterpret_cast<const float *>(old_w1->data().data());
    const float *b1 = reinterpret_cast<const float *>(old_b1->data().data());
    const float *w2 = reinterpret_cast<const float *>(old_w2->data().data());
    const float *b2 = reinterpret_cast<const float *>(old_b2->data().data());

    // new weights
    std::vector<float> weights(old_conv2->output_channels() * old_conv2->input_channels() * old_conv1->filter_h() * old_conv1->filter_w(), 0.f);
    for (size_t i = 0; i < weights.size(); i++)
    {
        auto m = i / (old_conv2->input_channels() * old_conv1->filter_h() * old_conv1->filter_w());
        auto remainder = i % (old_conv2->input_channels() * old_conv1->filter_h() * old_conv1->filter_w());
        auto c = remainder / (old_conv1->filter_h() * old_conv1->filter_w());
        remainder = remainder % (old_conv1->filter_h() * old_conv1->filter_w());
        auto r = remainder / old_conv1->filter_w();
        auto s = remainder % old_conv1->filter_w();
        weights[i] = w2[m * old_conv2->input_channels() + c] * w1[c * old_conv1->filter_h() * old_conv1->filter_w() + r * old_conv1->filter_w() + s];
    }
    auto new_weights = context.graph.emplace<constant>(dt_float32, shape_t { static_cast<unsigned long>(old_conv2->output_channels()), static_cast<unsigned long>(old_conv2->input_channels()), static_cast<unsigned long>(old_conv1->filter_h()), static_cast<unsigned long>(old_conv1->filter_w()) }, std::span(weights));
    new_weights->name(old_w1->name() + "_fuse_" + old_w2->name());

    // new bias
    std::vector<float> bias(old_conv2->output_channels(), 0.f);
    for (size_t m = 0; m < bias.size(); m++)
    {
        for (size_t c = 0; c < old_conv2->input_channels(); c++)
            bias[m] += b1[c] * w2[m * old_conv2->input_channels() + c];
        bias[m] += b2[m];
    }
    auto new_bias = context.graph.emplace<constant>(dt_float32, shape_t { static_cast<unsigned long>(old_conv2->output_channels()) }, std::span(bias));
    new_bias->name(old_b1->name() + "_fuse_" + old_b2->name());

    // create new conv2d
    auto new_conv = context.graph.emplace<conv2d>(old_conv1->input().shape(), new_weights->output().shape(), old_conv2->groups(), old_conv1->padding_h(), old_conv1->padding_w(), old_conv1->stride_h(), old_conv1->stride_w(), old_conv1->dilation_h(), old_conv1->dilation_w(), old_conv2->fused_activation());

    new_conv->input().connect(*context.inputs[0]->connection());
    new_conv->weights().connect(new_weights->output());
    new_conv->bias().connect(new_bias->output());
    new_conv->name(old_conv1->name() + "_fuse_" + old_conv2->name());

    auto inputs = context.outputs[0]->connections();
    for (auto &in : dup(inputs))
        in->connect(new_conv->output());
}

bool fold_two_1x1_conv2d_transform::on_try_match(node &node, transform_context &context)
{
    conv2d *conv1 = nullptr;
    conv2d *conv2 = nullptr;
    if ((conv1 = node_cast<conv2d>(node))
        && (conv2 = try_get_direct_child<conv2d>(*conv1))
        && (!conv1->is_depthwise() && !conv2->is_depthwise())
        && (conv1->filter_h() == 1 && conv1->filter_w() == 1)
        && (conv2->filter_h() == 1 && conv2->filter_w() == 1)
        && (conv1->groups() == 1 && conv2->groups() == 1)
        && (conv1->fused_activation() == value_range<float>::full())
        && conv1->outputs().size() == 1)
    {
        context.inputs.emplace_back(&conv1->input());
        context.outputs.emplace_back(&conv2->output());

        context.matched_nodes.emplace_back(conv1);
        context.matched_nodes.emplace_back(conv2);
        context.matched_nodes.emplace_back(&conv1->weights().connection()->owner());
        context.matched_nodes.emplace_back(&conv1->bias().connection()->owner());
        context.matched_nodes.emplace_back(&conv2->weights().connection()->owner());
        context.matched_nodes.emplace_back(&conv2->bias().connection()->owner());

        return true;
    }

    return false;
}

void fold_two_1x1_conv2d_transform::process([[maybe_unused]] transform_context &context)
{
    auto old_conv1 = node_cast<conv2d>(*context.matched_nodes[0]);
    auto old_conv2 = node_cast<conv2d>(*context.matched_nodes[1]);
    auto old_w1 = node_cast<constant>(*context.matched_nodes[2]);
    auto old_b1 = node_cast<constant>(*context.matched_nodes[3]);
    auto old_w2 = node_cast<constant>(*context.matched_nodes[4]);
    auto old_b2 = node_cast<constant>(*context.matched_nodes[5]);

    const float *w1 = reinterpret_cast<const float *>(old_w1->data().data());
    const float *b1 = reinterpret_cast<const float *>(old_b1->data().data());
    const float *w2 = reinterpret_cast<const float *>(old_w2->data().data());
    const float *b2 = reinterpret_cast<const float *>(old_b2->data().data());

    // new weights
    std::vector<float> weights(old_conv2->output_channels() * old_conv1->input_channels(), 0.f);
    for (size_t i = 0; i < weights.size(); i++)
    {
        auto m = i / old_conv1->input_channels();
        auto c = i % old_conv1->input_channels();
        for (auto k = 0; k < old_conv2->input_channels(); k++)
            weights[i] += w2[m * old_conv2->input_channels() + k] * w1[k * old_conv1->input_channels() + c];
    }
    auto new_weights = context.graph.emplace<constant>(dt_float32, shape_t { static_cast<unsigned long>(old_conv2->output_channels()), static_cast<unsigned long>(old_conv1->input_channels()), 1, 1 }, std::span(weights));
    new_weights->name(old_w1->name() + "_fuse_" + old_w2->name());

    // new bias
    std::vector<float> bias(old_conv2->output_channels(), 0.f);
    for (size_t m = 0; m < bias.size(); m++)
    {
        for (size_t c = 0; c < old_conv2->input_channels(); c++)
            bias[m] += b1[c] * w2[m * old_conv2->input_channels() + c];
        bias[m] += b2[m];
    }
    auto new_bias = context.graph.emplace<constant>(dt_float32, shape_t { static_cast<unsigned long>(old_conv2->output_channels()) }, std::span(bias));
    new_bias->name(old_b1->name() + "_fuse_" + old_b2->name());

    // create new conv2d
    auto new_conv = context.graph.emplace<conv2d>(old_conv1->input().shape(), new_weights->output().shape(), old_conv2->groups(), old_conv1->padding_h(), old_conv1->padding_w(), old_conv1->stride_h(), old_conv1->stride_w(), old_conv1->dilation_h(), old_conv1->dilation_w(), old_conv2->fused_activation());

    new_conv->input().connect(*context.inputs[0]->connection());
    new_conv->weights().connect(new_weights->output());
    new_conv->bias().connect(new_bias->output());
    new_conv->name(old_conv1->name() + "_fuse_" + old_conv2->name());

    auto inputs = context.outputs[0]->connections();
    for (auto &in : dup(inputs))
        in->connect(new_conv->output());
}