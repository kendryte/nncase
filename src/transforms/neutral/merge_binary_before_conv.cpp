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
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/merge_binary_before_conv.h>

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

std::vector<float> ToFloats(const std::vector<std::byte> &bytes)
{
    std::vector<float> floats(bytes.size() / sizeof(float), 0.0f);
    std::copy_n(bytes.begin(), floats.size() * sizeof(float),
        std::as_writable_bytes(std::span(floats)).begin());
    return floats;
}

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool split_binary_act_transform::on_try_match(node &node, transform_context &context)
{
    // constant *mul_const, *add_const;
    if (auto b_mul = node_cast<binary>(node))
    {
        if (b_mul->binary_op() == binary_mul)
        {
            if (try_get_direct_parent<conv2d>(*b_mul, 0) || try_get_direct_parent<conv2d>(*b_mul, 1))
                return false;
            if (auto tmp_binary = try_get_direct_parent<binary>(*b_mul, 0))
            {
                if (tmp_binary->binary_op() == binary_max)
                    return false;
            }
            if (auto b_add = try_get_direct_child<binary>(*b_mul))
            {
                if (b_add->output().connections().size() == 1 && b_add->output().connections()[0]->owner().runtime_opcode() == op_conv2d)
                {
                    if (b_add->binary_op() == binary_add && b_add->fused_activation().min == 0.f && b_add->fused_activation().max == std::numeric_limits<float>::infinity())
                    {
                        if (auto mul_const = try_get_direct_parent<constant>(*b_mul, 1))
                        {
                            context.inputs.emplace_back(&b_mul->input_a());
                            context.inputs.emplace_back(&b_mul->input_b());
                            context.matched_nodes.emplace_back(mul_const);
                        }
                        else if (auto mul_const = try_get_direct_parent<constant>(*b_mul, 0))
                        {
                            context.inputs.emplace_back(&b_mul->input_b());
                            context.inputs.emplace_back(&b_mul->input_a());
                            context.matched_nodes.emplace_back(mul_const);
                        }
                        else
                        {
                            return false;
                        }
                        if (auto add_const = try_get_direct_parent<constant>(*b_add, 1))
                        {
                            context.inputs.emplace_back(&b_mul->input_b());
                            context.matched_nodes.emplace_back(add_const);
                        }
                        else if (auto add_const = try_get_direct_parent<constant>(*b_add, 0))
                        {
                            context.inputs.emplace_back(&b_mul->input_a());
                            context.matched_nodes.emplace_back(add_const);
                        }
                        else
                        {
                            return false;
                        }

                        context.outputs.emplace_back(&b_add->output());
                        context.matched_nodes.emplace_back(b_mul);
                        context.matched_nodes.emplace_back(b_add);
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

void split_binary_act_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &const_mul = static_cast<constant &>(*context.matched_nodes[0]);
    auto &const_add = static_cast<constant &>(*context.matched_nodes[1]);
    // auto &b_mul = static_cast<binary &>(*context.matched_nodes[2]);
    auto &b_add = static_cast<binary &>(*context.matched_nodes[3]);

    std::vector<std::byte> mul_const_byte { const_mul.data().begin(), const_mul.data().end() };
    std::vector<std::byte> add_const_byte { const_add.data().begin(), const_add.data().end() };
    auto mul_const = ToFloats(mul_const_byte);
    auto add_const = ToFloats(add_const_byte);
    std::vector<float> max_const(output.shape()[1]);
    for (size_t i = 0; i < output.shape()[1]; i++)
    {
        max_const[i] = (0.f - static_cast<float>(add_const[i])) / static_cast<float>(mul_const[i]);
    }
    auto const_max = context.graph.emplace<constant>(dt_float32, shape_t { max_const.size(), 1, 1 }, max_const);
    auto b_max = context.graph.emplace<binary>(binary_max, output.type(), output.shape(), const_max->output().shape(), value_range<float>::full());
    const_max->name(b_add.name() + "_relu2max_const");
    b_max->name(b_add.name() + "_relu2max");
    b_max->input_a().connect(output);
    b_max->input_b().connect(const_max->output());
    // b_mul  input
    context.inputs[0]->connect(b_max->output());
    b_add.fused_activation().full(); // = value_range<float>::full();
    auto new_binary_add = context.graph.emplace<binary>(binary_add, b_add.output().type(), b_add.input_a().shape(), b_add.input_b().shape(), value_range<float>::full());
    new_binary_add->input_a().connect(b_add.input_a().connection()[0]);
    new_binary_add->input_b().connect(b_add.input_b().connection()[0]);
    new_binary_add->name(b_add.name());

    for (auto &in : dup(inputs))
    {
        in->connect(new_binary_add->output());
    }
}

bool merge_binary_b_into_conv_transform::on_try_match(node &node, transform_context &context)
{
    // constant *mul_const, *add_const;
    constant *weights = nullptr, *bias = nullptr;
    if (auto b_mul = node_cast<binary>(node))
    {
        if (b_mul->binary_op() == binary_mul)
        {
            if (try_get_direct_parent<conv2d>(*b_mul, 0) || try_get_direct_parent<conv2d>(*b_mul, 1))
                return false;
            if (try_get_direct_parent<binary>(*b_mul, 0))
            {
                if (auto b_add = try_get_direct_child<binary>(*b_mul))
                {
                    if (b_add->output().connections().size() == 1 && b_add->output().connections()[0]->owner().runtime_opcode() == op_conv2d)
                    {
                        if (b_add->binary_op() == binary_add && b_add->fused_activation() == value_range<float>::full())
                        {
                            if (auto mul_const = try_get_direct_parent<constant>(*b_mul, 1))
                            {
                                context.inputs.emplace_back(&b_mul->input_a());
                                context.matched_nodes.emplace_back(mul_const);
                            }
                            else if (auto mul_const = try_get_direct_parent<constant>(*b_mul, 0))
                            {
                                context.inputs.emplace_back(&b_mul->input_b());
                                context.matched_nodes.emplace_back(mul_const);
                            }
                            else
                            {
                                return false;
                            }
                            if (auto add_const = try_get_direct_parent<constant>(*b_add, 1))
                            {
                                context.matched_nodes.emplace_back(add_const);
                            }
                            else if (auto add_const = try_get_direct_parent<constant>(*b_add, 0))
                            {
                                context.matched_nodes.emplace_back(add_const);
                            }
                            else
                            {
                                return false;
                            }
                            if (auto conv = try_get_direct_child<conv2d>(*b_add))
                            {
                                if ((weights = try_get_direct_parent<constant>(*conv, 1))
                                    && (bias = try_get_direct_parent<constant>(*conv, 2)))
                                {
                                    context.matched_nodes.emplace_back(weights);
                                    context.matched_nodes.emplace_back(bias);
                                    context.matched_nodes.emplace_back(conv);
                                }
                                context.outputs.emplace_back(&conv->output());
                            }

                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

void merge_binary_b_into_conv_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &const_mul = static_cast<constant &>(*context.matched_nodes[0]);
    auto &const_add = static_cast<constant &>(*context.matched_nodes[1]);
    auto &weights = static_cast<constant &>(*context.matched_nodes[2]);
    auto &bias = static_cast<constant &>(*context.matched_nodes[3]);
    auto &conv = static_cast<conv2d &>(*context.matched_nodes[4]);

    std::vector<std::byte> weights_byte { weights.data().begin(), weights.data().end() };
    auto weights_const = ToFloats(weights_byte);
    auto bitc_mul_const = context.graph.emplace<bitcast>(dt_float32, const_mul.output().shape(), shape_t { 1, const_mul.output().shape()[0], 1, 1 });
    bitc_mul_const->name(conv.name() + "bitc_mul_const");
    auto new_weights = context.graph.emplace<binary>(binary_mul, weights.output().type(), weights.output().shape(), bitc_mul_const->output().shape(), value_range<float>::full());
    new_weights->name(conv.name() + "binary_mul_weights");
    bitc_mul_const->input().connect(const_mul.output());
    new_weights->input_a().connect(weights.output());
    new_weights->input_b().connect(bitc_mul_const->output());

    auto bitc_add_const = context.graph.emplace<bitcast>(dt_float32, const_add.output().shape(), shape_t { 1, const_add.output().shape()[0], 1, 1 });
    bitc_add_const->name(conv.name() + "bitc_add_const");
    bitc_add_const->input().connect(const_add.output());

    auto add_weights = context.graph.emplace<constant>(dt_float32, weights.output().shape(), weights_const);
    add_weights->name(conv.name() + "add_weights");

    std::vector<float> zeros_const(weights.output().shape()[0], 0.f);
    auto add_conv_bias = context.graph.emplace<constant>(dt_float32, shape_t { weights.output().shape()[0] }, zeros_const);
    add_conv_bias->name(conv.name() + "add_conv_bias");

    auto add_conv = context.graph.emplace<conv2d>(bitc_add_const->output().shape(), add_weights->output().shape(), conv.groups(),
        conv.padding_h(), conv.padding_w(), conv.stride_h(), conv.stride_w(), conv.dilation_h(), conv.dilation_w(), value_range<float>::full());
    add_conv->name(conv.name() + "add_conv_weights");
    add_conv->input().connect(bitc_add_const->output());
    add_conv->weights().connect(add_weights->output());
    add_conv->bias().connect(add_conv_bias->output());

    auto bitc_bias = context.graph.emplace<bitcast>(dt_float32, bias.output().shape(), shape_t { 1, bias.output().shape()[0], 1, 1 });
    bitc_bias->name(conv.name() + "bitc_bias");
    bitc_bias->input().connect(bias.output());

    auto add_bias = context.graph.emplace<binary>(binary_add, add_conv->output().type(), add_conv->output().shape(), bitc_bias->output().shape(), value_range<float>::full());
    add_bias->name(conv.name() + "add_bias");
    add_bias->input_a().connect(add_conv->output());
    add_bias->input_b().connect(bitc_bias->output());

    auto bitc_post = context.graph.emplace<bitcast>(dt_float32, add_bias->output().shape(), shape_t { bias.output().shape()[0] });
    bitc_post->name(conv.name() + "bitc_post");
    bitc_post->input().connect(add_bias->output());

    auto new_conv = context.graph.emplace<conv2d>(output.shape(), new_weights->output().shape(), conv.groups(),
        conv.padding_h(), conv.padding_w(), conv.stride_h(), conv.stride_w(), conv.dilation_h(), conv.dilation_w(), conv.fused_activation());

    new_conv->bias().connect(bitc_post->output());
    new_conv->weights().connect(new_weights->output());
    new_conv->input().connect(output);
    new_conv->name(conv.name() + "new_conv");
    for (auto &in : dup(inputs))
    {
        in->connect(new_conv->output());
    }
}
