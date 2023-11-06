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
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/reduce_window2d.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/binary_motion.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool binary_reduce_window2d_motion_up_transform::on_try_match(node &node, transform_context &context)
{
    conv2d *conv;
    reduce_window2d *pool;
    constant *c;
    binary *b;
    if ((conv = node_cast<conv2d>(node))
        && (pool = try_get_direct_child<reduce_window2d>(*conv))
        && (b = try_get_direct_child<binary>(*pool))
        && (c = try_get_direct_parent<constant>(*b)))
    {
        auto &c_shape = c->output().shape();
        if (conv->fused_activation() == value_range<float>::full()
            && pool->fused_activation() == value_range<float>::full()
            && (pool->reduce_op() == reduce_max
                || pool->reduce_op() == reduce_min)
            && (b->binary_op() == binary_add
                || b->binary_op() == binary_mul)
            && ((c_shape.size() == 3
                    && c_shape == shape_t { (size_t)conv->output_channels(), 1, 1 })
                || (c_shape.size() == 4
                    && c_shape == shape_t { 1, (size_t)conv->output_channels(), 1, 1 })))
        {
            auto data = as_span<const float>(c->data());
            if (b->binary_op() == binary_mul
                && std::any_of(data.begin(), data.end(), [](float v) { return v < 0.f; }))
                return false;

            context.matched_nodes.emplace_back(conv);
            context.matched_nodes.emplace_back(pool);
            context.matched_nodes.emplace_back(c);
            context.matched_nodes.emplace_back(b);

            context.inputs.emplace_back(&conv->input());
            context.inputs.emplace_back(&conv->weights());
            context.inputs.emplace_back(&conv->bias());
            context.outputs.emplace_back(&b->output());

            return true;
        }
    }

    return false;
}

void binary_reduce_window2d_motion_up_transform::process(transform_context &context)
{
    auto inputs = context.outputs[0]->connections();

    auto &conv = static_cast<conv2d &>(*context.matched_nodes[0]);
    auto &pool = static_cast<reduce_window2d &>(*context.matched_nodes[1]);
    auto &c = static_cast<constant &>(*context.matched_nodes[2]);
    auto &old_b = static_cast<binary &>(*context.matched_nodes[3]);

    auto b = context.graph.emplace<binary>(old_b.binary_op(), conv.output().type(), conv.output().shape(), c.output().shape(), old_b.fused_activation());
    b->attributes(old_b.attributes());
    b->name(old_b.name());
    b->input_a().connect(conv.output());
    b->input_b().connect(c.output());
    pool.input().connect(b->output());

    for (auto &in : dup(inputs))
        in->connect(pool.output());
}
