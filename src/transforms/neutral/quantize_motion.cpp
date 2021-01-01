/* Copyright 2020 Canaan Inc.
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
#include <nncase/ir/ops/batch_to_space.h>
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/quantize.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/resize_image.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/ops/transpose.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/quantize_motion.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool quantize_pad_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto q = node_cast<quantize>(node))
    {
        if (auto p = try_get_direct_parent<pad>(*q))
        {
            context.matched_nodes.emplace_back(q);
            context.matched_nodes.emplace_back(p);

            context.inputs.emplace_back(&p->input());
            context.outputs.emplace_back(&q->output());

            return false;
        }
    }

    return false;
}
/**
 *         pad                      quantize
 *         |                        |
 *         |                        |
 *         quantize     -->         pad
 **/
void quantize_pad_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_q = static_cast<quantize &>(*context.matched_nodes[0]);
    auto &old_p = static_cast<pad &>(*context.matched_nodes[1]);
    auto q_param = old_q.quant_param();
    assert(q_param.zero_point.size() == 1); // TODO: supprot quant by channel

    //dequantize pad_value when pad move down quantize
    auto pad_value = std::clamp(
        (int32_t)std::round((old_p.pad_value().as<float>() - q_param.zero_point[0]) * q_param.scale[0]),
        0, 255);

    auto q = context.graph.emplace<quantize>(old_p.input().shape(), old_p.input().type(), q_param);
    q->name(old_q.name());
    auto p = context.graph.emplace<pad>(q->output().type(), q->output().shape(), old_p.paddings(), (float_t)pad_value);
    p->name(old_p.name());

    p->input().connect(q->output());
    q->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(p->output());
}

bool quantize_transpose_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto q = node_cast<quantize>(node))
    {
        if (auto tp = try_get_direct_parent<transpose>(*q))
        {
            context.matched_nodes.emplace_back(tp);
            context.matched_nodes.emplace_back(q);

            context.inputs.emplace_back(&tp->input());
            context.outputs.emplace_back(&q->output());

            return true;
        }
    }

    return false;
}
/**
 *         tp                       quantize
 *         |                        |
 *         |                        |
 *         quantize     -->         tp
 **/
void quantize_transpose_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_tp = static_cast<transpose &>(*context.matched_nodes[0]);
    auto &old_q = static_cast<quantize &>(*context.matched_nodes[1]);

    auto q = context.graph.emplace<quantize>(old_tp.input().shape(), old_q.output().type(), old_q.quant_param());
    q->name(old_q.name());
    auto tp = context.graph.emplace<transpose>(q->output().type(), q->output().shape(), old_tp.perm());
    tp->name(old_tp.name());

    tp->input().connect(q->output());

    q->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(tp->output());
}

bool quantize_slice_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto q = node_cast<quantize>(node))
    {
        if (auto sl = try_get_direct_parent<slice>(*q))
        {
            if (auto r = try_get_direct_parent<bitcast>(*sl))
            {
                if (auto tp = try_get_direct_parent<transpose>(*r))
                {
                    if (try_get_direct_parent<bitcast>(*tp))
                    {
                        //if meet b2s, do nothing
                        return false;
                    }
                }
            }
            context.matched_nodes.emplace_back(q);
            context.matched_nodes.emplace_back(sl);

            context.inputs.emplace_back(&sl->input());
            context.outputs.emplace_back(&q->output());

            return true;
        }
    }

    return false;
}
/**
 *         slice                  quantize
 *         |                        |
 *         |                        |
 *         quantize     -->         slice
 **/
void quantize_slice_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_q = static_cast<quantize &>(*context.matched_nodes[0]);
    auto &old_slice = static_cast<slice &>(*context.matched_nodes[1]);

    auto q = context.graph.emplace<quantize>(old_slice.input().shape(), old_q.output().type(), old_q.quant_param());
    q->name(old_q.name());
    auto sl = context.graph.emplace<slice>(q->output().type(), q->output().shape(), old_slice.begin(), old_slice.end(), old_slice.strides(),
        old_slice.begin_mask(), old_slice.end_mask(), old_slice.ellipsis_mask(), old_slice.new_axis_mask(), old_slice.shrink_axis_mask());
    sl->name(old_slice.name());

    q->input().connect(output);
    sl->input().connect(q->output());
    for (auto &in : dup(inputs))
        in->connect(sl->output());
}

bool quantize_resize_image_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto q = node_cast<quantize>(node))
    {
        if (auto resize = try_get_direct_parent<resize_image>(*q))
        {
            context.matched_nodes.emplace_back(q);
            context.matched_nodes.emplace_back(resize);

            context.inputs.emplace_back(&resize->input());
            context.outputs.emplace_back(&q->output());

            return true;
        }
    }

    return false;
}
/**
 *         resize                 quantize
 *         |                        |
 *         |                        |
 *         quantize     -->         resize
 **/
void quantize_resize_image_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_q = static_cast<quantize &>(*context.matched_nodes[0]);
    auto &old_resize = static_cast<resize_image &>(*context.matched_nodes[1]);

    auto q = context.graph.emplace<quantize>(old_resize.input().shape(), old_q.output().type(), old_q.quant_param());
    q->name(old_q.name());
    auto resize = context.graph.emplace<resize_image>(q->output().type(), old_resize.mode(), q->output().shape(), old_resize.new_size(), old_resize.align_corners());
    resize->name(old_resize.name());

    q->input().connect(output);
    resize->input().connect(q->output());
    for (auto &in : dup(inputs))
        in->connect(resize->output());
}

bool quantize_reshape_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto q = node_cast<quantize>(node))
    {
        if (auto r = try_get_direct_parent<bitcast>(*q))
        {
            if (auto tp = try_get_direct_parent<transpose>(*r))
            {
                if (auto r2 = try_get_direct_parent<bitcast>(*tp))
                {
                    if (try_get_direct_parent<pad>(*r2))
                    {
                        //if meet s2b, do nothing
                        return false;
                    }
                }
            }
            context.matched_nodes.emplace_back(q);
            context.matched_nodes.emplace_back(r);

            context.inputs.emplace_back(&r->input());
            context.outputs.emplace_back(&q->output());

            return true;
        }
    }
    return false;
}
/**
 *         reshape                  quantize
 *         |                        |
 *         |                        |
 *         quantize     -->         reshape
 **/
void quantize_reshape_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_q = static_cast<quantize &>(*context.matched_nodes[0]);
    auto &old_r = static_cast<bitcast &>(*context.matched_nodes[1]);

    auto q = context.graph.emplace<quantize>(old_r.input().shape(), old_q.output().type(), old_q.quant_param());
    q->name(old_q.name());
    auto r = context.graph.emplace<bitcast>(q->output().type(), q->output().shape(), q->output().type(), old_r.new_shape());
    r->name(old_r.name());

    q->input().connect(output);
    r->input().connect(q->output());

    for (auto &in : dup(inputs))
        in->connect(r->output());
}

bool quantize_bitcast_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto q = node_cast<quantize>(node))
    {
        if (auto bc = try_get_direct_parent<bitcast>(*q))
        {
            context.matched_nodes.emplace_back(q);
            context.matched_nodes.emplace_back(bc);

            context.inputs.emplace_back(&bc->input());
            context.outputs.emplace_back(&q->output());

            return true;
        }
    }

    return false;
}
/**
 *         bc                       quantize
 *         |                        |
 *         |                        |
 *         quantize     -->         bc
 **/
void quantize_bitcast_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_q = static_cast<quantize &>(*context.matched_nodes[0]);
    auto &old_bc = static_cast<bitcast &>(*context.matched_nodes[1]);

    auto q = context.graph.emplace<quantize>(old_bc.input().shape(), old_q.output().type(), old_q.quant_param());
    q->name(old_q.name());
    auto bc = context.graph.emplace<bitcast>(q->output().type(), q->output().shape(), old_q.output().type(), old_q.output().shape());
    bc->name(old_bc.name());

    bc->input().connect(q->output());

    q->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(bc->output());
}

bool quantize_b2s_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto q = node_cast<quantize>(node))
    {
        if (auto b2s = try_get_direct_child<batch_to_space>(*q))
        {
            context.matched_nodes.emplace_back(q);
            context.matched_nodes.emplace_back(b2s);

            context.inputs.emplace_back(&q->input());
            context.outputs.emplace_back(&b2s->output());

            return true;
        }
    }

    return false;
}

void quantize_b2s_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_q = static_cast<quantize &>(*context.matched_nodes[0]);
    auto &old_b2s = static_cast<batch_to_space &>(*context.matched_nodes[1]);

    auto b2s = context.graph.emplace<batch_to_space>(output.type(), output.shape(), old_b2s.block_size_h(), old_b2s.block_size_w(),
        old_b2s.strides(), old_b2s.begin(), old_b2s.end(), old_b2s.crop_h(), old_b2s.crop_w());
    b2s->name(old_b2s.name());
    auto q = context.graph.emplace<quantize>(b2s->output().shape(), old_q.output().type(), old_q.quant_param());
    q->name(old_q.name());

    b2s->input().connect(output);
    q->input().connect(b2s->output());
    for (auto &in : dup(inputs))
        in->connect(q->output());
}