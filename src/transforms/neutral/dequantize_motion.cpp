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
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/dequantize.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/resize_image.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/ops/space_to_batch.h>
#include <nncase/ir/ops/transpose.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/dequantize_motion.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool dequantize_pad_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto deq = node_cast<dequantize>(node))
    {
        if (auto p = try_get_direct_parent<pad>(*deq))
        {
            context.matched_nodes.emplace_back(p);
            context.matched_nodes.emplace_back(deq);

            context.inputs.emplace_back(&p->input());
            context.outputs.emplace_back(&deq->output());

            return true;
        }
    }

    return false;
}

void dequantize_pad_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_p = static_cast<pad &>(*context.matched_nodes[0]);
    auto &old_deq = static_cast<dequantize &>(*context.matched_nodes[1]);
    auto q_param = old_deq.quant_param();
    auto pad_value = q_param.zero_point;

    auto deq = context.graph.emplace<dequantize>(old_p.input().type(), old_p.input().shape(), old_deq.output().type(), q_param);
    deq->name(old_deq.name());
    auto p = context.graph.emplace<pad>(deq->output().type(), deq->output().shape(), old_p.paddings(), old_p.pad_mode(), (uint8_t)pad_value);
    p->name(old_p.name());
    p->input().connect(deq->output());

    deq->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(p->output());
}

bool dequantize_transpose_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto deq = node_cast<dequantize>(node))
    {
        if (auto tp = try_get_direct_child<transpose>(*deq))
        {
            /**
             * transpose 和量化节点优化的策略还是改成了遇到dequantize时transpose取消下移
             * 将两个OP之间可以进行优化的操作尽量放在量化后 反量化前
             * 同时增加了transpose 可以穿过反量化节点&binary同时出现的这种特例
             * 在输入为NCHW且对输入增加了transpose,输入size*size>2^16时 broadcast会报错
             * **/
            if (try_get_direct_child<binary>(*tp))
                return false;
            context.matched_nodes.emplace_back(deq);
            context.matched_nodes.emplace_back(tp);

            context.inputs.emplace_back(&deq->input());
            context.outputs.emplace_back(&tp->output());

            return true;
        }
    }

    return false;
}

void dequantize_transpose_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_deq = static_cast<dequantize &>(*context.matched_nodes[0]);
    auto &old_tp = static_cast<transpose &>(*context.matched_nodes[1]);

    auto tp = context.graph.emplace<transpose>(old_deq.input().type(), old_deq.input().shape(), old_tp.perm());
    tp->name(old_tp.name());
    auto deq = context.graph.emplace<dequantize>(tp->output().type(), tp->output().shape(), old_deq.output().type(), old_deq.quant_param());
    deq->name(old_deq.name());

    deq->input().connect(tp->output());
    tp->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}

/**
 *      tp                  deq
 *      |                    |
 *      deq         ->      tp
 *      |                    |
 *      binary              binary
 **/
bool dequantize_transbin_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto deq = node_cast<dequantize>(node))
    {
        if (auto tp = try_get_direct_parent<transpose>(*deq))
        {
            if (try_get_direct_child<binary>(*deq))
            {
                context.matched_nodes.emplace_back(tp);
                context.matched_nodes.emplace_back(deq);

                context.inputs.emplace_back(&tp->input());
                context.outputs.emplace_back(&deq->output());
                return true;
            }
        }
    }

    return false;
}

void dequantize_transbin_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_tp = static_cast<transpose &>(*context.matched_nodes[0]);
    auto &old_deq = static_cast<dequantize &>(*context.matched_nodes[1]);

    auto deq = context.graph.emplace<dequantize>(old_tp.input().type(), old_tp.input().shape(), old_deq.output().type(), old_deq.quant_param());
    deq->name(old_deq.name());
    auto tp = context.graph.emplace<transpose>(deq->output().type(), deq->output().shape(), old_tp.perm());
    tp->name(old_tp.name());

    deq->input().connect(output);
    tp->input().connect(deq->output());
    for (auto &in : dup(inputs))
        in->connect(tp->output());
}

bool dequantize_slice_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto deq = node_cast<dequantize>(node))
    {
        for (auto &out : deq->output().connections())
        {
            context.matched_nodes.emplace_back(deq);
            context.inputs.emplace_back(&deq->input());
            if (out->owner().runtime_opcode() == op_slice)
            {
                auto slice_node = node_cast<slice>(out->owner());
                context.matched_nodes.emplace_back(slice_node);
                context.outputs.emplace_back(&slice_node->output());
            }
            else
            {
                return false;
            }
            return true;
        }
    }

    return false;
}

void dequantize_slice_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto &old_deq = static_cast<dequantize &>(*context.matched_nodes[0]);

    for (int i = 1; i < context.matched_nodes.size(); i++)
    {
        auto inputs = context.outputs[i - 1]->connections();
        auto &old_slice = static_cast<slice &>(*context.matched_nodes[i]);
        auto sl = context.graph.emplace<slice>(old_deq.input().type(), old_deq.input().shape(), old_slice.begin(), old_slice.end(), old_slice.strides(),
            old_slice.begin_mask(), old_slice.end_mask(), old_slice.ellipsis_mask(), old_slice.new_axis_mask(), old_slice.shrink_axis_mask());
        sl->name(old_slice.name());
        auto deq = context.graph.emplace<dequantize>(sl->output().type(), sl->output().shape(), old_deq.output().type(), old_deq.quant_param());
        deq->name(old_deq.name());
        deq->input().connect(sl->output());
        sl->input().connect(output);
        for (auto &in : dup(inputs))
            in->connect(deq->output());
    }
}

bool dequantize_resize_image_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto deq = node_cast<dequantize>(node))
    {
        if (auto resize = try_get_direct_child<resize_image>(*deq))
        {
            context.matched_nodes.emplace_back(deq);
            context.matched_nodes.emplace_back(resize);

            context.inputs.emplace_back(&deq->input());
            context.outputs.emplace_back(&resize->output());

            return true;
        }
    }

    return false;
}

void dequantize_resize_image_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_deq = static_cast<dequantize &>(*context.matched_nodes[0]);
    auto &old_resize = static_cast<resize_image &>(*context.matched_nodes[1]);

    auto resize = context.graph.emplace<resize_image>(dt_uint8, old_resize.mode(), old_deq.input().shape(), old_resize.new_size(), old_resize.align_corners());
    resize->name(old_resize.name());
    auto deq = context.graph.emplace<dequantize>(resize->output().type(), resize->output().shape(), old_deq.output().type(), old_deq.quant_param());
    deq->name(old_deq.name());
    deq->input().connect(resize->output());

    resize->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}

bool dequantize_reshape_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto deq = node_cast<dequantize>(node))
    {
        if (auto r = try_get_direct_child<bitcast>(*deq))
        {
            if (auto tp = try_get_direct_child<transpose>(*r))
            {
                if (auto r2 = try_get_direct_child<bitcast>(*tp))
                {
                    if (try_get_direct_child<slice>(*r2))
                    {
                        //if meet b2s, do nothing
                        return false;
                    }
                }
            }
            context.matched_nodes.emplace_back(deq);
            context.matched_nodes.emplace_back(r);

            context.inputs.emplace_back(&deq->input());
            context.outputs.emplace_back(&r->output());

            return true;
        }
    }

    return false;
}

void dequantize_reshape_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_deq = static_cast<dequantize &>(*context.matched_nodes[0]);
    auto &old_r = static_cast<bitcast &>(*context.matched_nodes[1]);

    auto r = context.graph.emplace<bitcast>(dt_uint8, old_deq.input().shape(), dt_uint8, old_r.new_shape());
    r->name(old_r.name());
    auto deq = context.graph.emplace<dequantize>(r->output().type(), r->output().shape(), old_deq.output().type(), old_deq.quant_param());
    deq->name(old_deq.name());
    deq->input().connect(r->output());

    r->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}

bool dequantize_bitcast_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto deq = node_cast<dequantize>(node))
    {
        if (auto bc = try_get_direct_child<bitcast>(*deq))
        {
            context.matched_nodes.emplace_back(deq);
            context.matched_nodes.emplace_back(bc);

            context.inputs.emplace_back(&deq->input());
            context.outputs.emplace_back(&bc->output());

            return true;
        }
    }

    return false;
}
/**
 *         bc                       bc
 *         |                        |
 *         |                        |
 *         dequantize     -->       bc2
 *         |                        |
 *         |                        |
 *         bc2                      dequantize
 **/
void dequantize_bitcast_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_deq = static_cast<dequantize &>(*context.matched_nodes[0]);
    auto &old_bc = static_cast<bitcast &>(*context.matched_nodes[1]);

    auto bc = context.graph.emplace<bitcast>(old_deq.input().type(), old_deq.input().shape(), old_deq.input().type(), old_bc.output().shape());
    bc->name(old_bc.name());
    auto deq = context.graph.emplace<dequantize>(bc->output().type(), bc->output().shape(), old_deq.output().type(), old_deq.quant_param());
    deq->name(old_deq.name());
    link(old_bc.output(), deq->output(), context.quantizer);
    deq->input().connect(bc->output());

    bc->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}

bool dequantize_s2b_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto deq = node_cast<dequantize>(node))
    {
        if (auto s2b = try_get_direct_parent<space_to_batch>(*deq))
        {
            context.matched_nodes.emplace_back(s2b);
            context.matched_nodes.emplace_back(deq);

            context.inputs.emplace_back(&s2b->input());
            context.outputs.emplace_back(&deq->output());

            return true;
        }
    }

    return false;
}

void dequantize_s2b_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_s2b = static_cast<space_to_batch &>(*context.matched_nodes[0]);
    auto &old_deq = static_cast<dequantize &>(*context.matched_nodes[1]);

    auto deq = context.graph.emplace<dequantize>(output.type(), output.shape(), old_deq.output().type(), old_deq.quant_param());
    deq->name(old_deq.name());
    auto s2b = context.graph.emplace<space_to_batch>(deq->output().type(), deq->output().shape(), old_s2b.block_size_h(), old_s2b.block_size_w(),
        old_s2b.padding_h(), old_s2b.padding_w(), old_s2b.pad_value(), old_s2b.real_block_size_h(), old_s2b.real_block_size_w());
    s2b->name(old_s2b.name());

    deq->input().connect(output);
    s2b->input().connect(deq->output());
    for (auto &in : dup(inputs))
        in->connect(s2b->output());
}