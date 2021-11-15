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
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/fold_binary_into_conv.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

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

bool fold_add_before_conv_transform::on_try_match(node &node, transform_context &context)
{
    if (auto b = node_cast<binary>(node))
    {
        if (b->binary_op() != binary_add)
            return false;
        if (b->fused_activation() != value_range<float>::full())
        {
            return false;
        }
        if (auto conv = try_get_direct_child<conv2d>(*b))
        {
            if (auto b_const = try_get_direct_parent<constant>(*b, 1))
            {
                context.inputs.emplace_back(&b->input_a());
                context.matched_nodes.emplace_back(b_const);
            }
            else if (auto b_const = try_get_direct_parent<constant>(*b, 0))
            {
                context.inputs.emplace_back(&b->input_b());
                context.matched_nodes.emplace_back(b_const);
            }
            context.outputs.emplace_back(&conv->output());

            context.matched_nodes.emplace_back(b);
            context.matched_nodes.emplace_back(conv);

            return true;
        }
    }

    return false;
}

void fold_add_before_conv_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &binary_c = *static_cast<constant *>(context.matched_nodes[0]);
    auto &conv = *static_cast<conv2d *>(context.matched_nodes[2]);

    std::vector<std::byte> binary_c_byte { binary_c.data().begin(), binary_c.data().end() };
    auto binary_const = ToFloats(binary_c_byte);
    if (binary_const.size() == 1)
    {
        for (size_t i = 1; i < output.shape()[1]; i++)
            binary_const.emplace_back(binary_const[0]);
    }

    std::vector<float> binary_conv_const_input(xt::compute_size(output.shape()), 0.f);
    for (size_t n = 0; n < output.shape()[0]; n++)
    {
        for (size_t c = 0; c < output.shape()[1]; c++)
        {
            for (size_t h = 0; h < output.shape()[2]; h++)
            {
                for (size_t w = 0; w < output.shape()[3]; w++)
                {
                    binary_conv_const_input[n * output.shape()[1] * output.shape()[2] * output.shape()[3] + c * output.shape()[2] * output.shape()[3] + h * output.shape()[3] + w] = binary_const[c];
                }
            }
        }
    }
    auto binary_conv_input = context.graph.emplace<constant>(dt_float32, output.shape(), binary_conv_const_input);

    auto old_weights = node_cast<constant>(conv.input_at(1).connection()->owner());
    std::vector<std::byte> new_weights_byte { old_weights->data().begin(), old_weights->data().end() };
    auto new_weights = ToFloats(new_weights_byte);
    auto binary_conv_weights = context.graph.emplace<constant>(dt_float32, conv.weights().shape(), new_weights);

    std::vector<float> new_bias_const(conv.bias().shape()[0], 0.f);
    auto new_bias = context.graph.emplace<constant>(dt_float32, conv.bias().shape(), new_bias_const);

    auto new_conv = context.graph.emplace<conv2d>(binary_conv_input->output().shape(), binary_conv_weights->output().shape(),
        conv.groups(), conv.padding_h(), conv.padding_w(), conv.stride_h(), conv.stride_w(),
        conv.dilation_h(), conv.dilation_w(), conv.fused_activation());
    new_conv->name(conv.name() + "/new_bias_conv");
    binary_conv_input->name(new_conv->name() + "_input");
    binary_conv_weights->name(new_conv->name() + "_weights");
    new_bias->name(new_conv->name() + "_bias");

    new_conv->input().connect(binary_conv_input->output());
    new_conv->weights().connect(binary_conv_weights->output());
    new_conv->bias().connect(new_bias->output());

    conv.input().connect(output);
    auto source_conv = context.graph.emplace<conv2d>(output.shape(), conv.weights().shape(),
        conv.groups(), conv.padding_h(), conv.padding_w(), conv.stride_h(), conv.stride_w(),
        conv.dilation_h(), conv.dilation_w(), conv.fused_activation());
    source_conv->name(conv.name());
    source_conv->input().connect(output);
    source_conv->weights().connect(conv.weights().connection()[0]);
    source_conv->bias().connect(conv.bias().connection()[0]);

    auto bias_add = context.graph.emplace<binary>(binary_add, new_conv->output().shape(), source_conv->output().shape(), value_range<float>::full());
    bias_add->name(conv.name() + "_new_add");
    bias_add->input_b().connect(new_conv->output());
    bias_add->input_a().connect(source_conv->output());

    for (auto &in : dup(inputs))
    {
        in->connect(bias_add->output());
    }
}

bool fold_mul_before_conv_transform::on_try_match(node &node, transform_context &context)
{
    if (auto b = node_cast<binary>(node))
    {
        if (b->binary_op() != binary_mul)
            return false;
        if (b->fused_activation() != value_range<float>::full())
        {
            return false;
        }
        if (auto conv = try_get_direct_child<conv2d>(*b))
        {
            if (auto b_const = try_get_direct_parent<constant>(*b, 1))
            {
                context.inputs.emplace_back(&b->input_a());
                context.matched_nodes.emplace_back(b_const);
            }
            else if (auto b_const = try_get_direct_parent<constant>(*b, 0))
            {
                context.inputs.emplace_back(&b->input_b());
                context.matched_nodes.emplace_back(b_const);
            }
            context.matched_nodes.emplace_back(b);
            context.matched_nodes.emplace_back(conv);

            return true;
        }
    }

    return false;
}

void fold_mul_before_conv_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();

    auto &binary_c = *static_cast<constant *>(context.matched_nodes[0]);
    auto &conv = *static_cast<conv2d *>(context.matched_nodes[2]);

    auto old_weights = node_cast<constant>(conv.input_at(1).connection()->owner());
    std::vector<std::byte> binary_c_byte { binary_c.data().begin(), binary_c.data().end() };
    auto binary_const = ToFloats(binary_c_byte);
    if (binary_const.size() == 1)
    {
        for (size_t i = 1; i < old_weights->output().shape()[1]; i++)
            binary_const.emplace_back(binary_const[0]);
    }
    auto mul_constant = context.graph.emplace<constant>(dt_float32, shape_t { 1, old_weights->output().shape()[1], 1, 1 }, binary_const);
    mul_constant->name(binary_c.name() + "/mul_const");
    auto b_mul_weights = context.graph.emplace<binary>(binary_mul, old_weights->output().shape(), mul_constant->output().shape(), value_range<float>::full());
    b_mul_weights->name(conv.name() + "/mul_weights");
    b_mul_weights->input_a().connect(old_weights->output());
    b_mul_weights->input_b().connect(mul_constant->output());

    conv.weights().connect(b_mul_weights->output());
    conv.input().connect(output);
}