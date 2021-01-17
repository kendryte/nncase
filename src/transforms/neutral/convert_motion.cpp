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
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/convert.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/resize_image.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/ops/transpose.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/convert_motion.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool convert_bitcast_motion_up_transform::on_try_match(node &node, transform_context &context)
{
    if (auto r = node_cast<bitcast>(node))
    {
        if (auto cvt = try_get_direct_child<convert>(*r))
        {
            if (ir::get_bytes(cvt->input().type()) > ir::get_bytes(cvt->new_type())
                && r->is_reshape())
            {
                context.matched_nodes.emplace_back(r);
                context.matched_nodes.emplace_back(cvt);

                context.inputs.emplace_back(&r->input());
                context.outputs.emplace_back(&cvt->output());

                return true;
            }
        }
    }

    return false;
}

void convert_bitcast_motion_up_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_r = static_cast<bitcast &>(*context.matched_nodes[0]);
    auto &old_cvt = static_cast<convert &>(*context.matched_nodes[1]);

    auto cvt = context.graph.emplace<convert>(output.type(), output.shape(), old_cvt.new_type());
    cvt->name(old_cvt.name());
    auto r = context.graph.emplace<bitcast>(cvt->output().type(), cvt->output().shape(), old_r.new_shape());
    r->name(old_r.name());
    r->input().connect(cvt->output());

    cvt->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(r->output());
}

bool convert_bitcast_motion_down_transform::on_try_match(node &node, transform_context &context)
{
    if (auto cvt = node_cast<convert>(node))
    {
        if (auto r = try_get_direct_child<bitcast>(*cvt))
        {
            if (ir::get_bytes(cvt->input().type()) < ir::get_bytes(cvt->new_type())
                && r->is_reshape())
            {
                context.matched_nodes.emplace_back(cvt);
                context.matched_nodes.emplace_back(r);

                context.inputs.emplace_back(&cvt->input());
                context.outputs.emplace_back(&r->output());

                return true;
            }
        }
    }

    return false;
}

void convert_bitcast_motion_down_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_cvt = static_cast<convert &>(*context.matched_nodes[0]);
    auto &old_r = static_cast<bitcast &>(*context.matched_nodes[1]);

    auto r = context.graph.emplace<bitcast>(output.type(), old_cvt.input().shape(), old_r.new_shape());
    r->name(old_r.name());
    auto cvt = context.graph.emplace<convert>(r->output().type(), r->output().shape(), old_cvt.new_type());
    cvt->name(old_cvt.name());
    cvt->input().connect(r->output());

    r->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(cvt->output());
}

bool convert_concat_motion_up_transform::on_try_match(node &node, transform_context &context)
{
    if (auto c = node_cast<concat>(node))
    {
        if (auto cvt = try_get_direct_child<convert>(*c))
        {
            if (ir::get_bytes(cvt->input().type()) > ir::get_bytes(cvt->new_type()))
            {
                context.matched_nodes.emplace_back(c);
                context.matched_nodes.emplace_back(cvt);

                for (auto in : c->inputs())
                    context.inputs.emplace_back(in);
                context.outputs.emplace_back(&cvt->output());

                return true;
            }
        }
    }

    return false;
}

void convert_concat_motion_up_transform::process(transform_context &context)
{
    auto inputs = context.outputs[0]->connections();

    auto &old_c = static_cast<concat &>(*context.matched_nodes[0]);
    auto &old_cvt = static_cast<convert &>(*context.matched_nodes[1]);

    std::vector<convert *> new_converts;
    new_converts.reserve(context.inputs.size());
    for (auto &in : context.inputs)
    {
        auto &output = *in->connection();
        auto cvt = context.graph.emplace<convert>(output.type(), output.shape(), old_cvt.new_type());
        cvt->name(old_cvt.name());
        cvt->input().connect(output);
        new_converts.emplace_back(cvt);
    }

    auto input_shapes = ir::get_input_shapes(old_c.inputs());
    auto c = context.graph.emplace<concat>(old_cvt.output().type(), input_shapes, old_c.axis());
    c->name(old_c.name());
    for (size_t i = 0; i < new_converts.size(); i++)
        c->input_at(i).connect(new_converts.at(i)->output());

    for (auto &in : dup(inputs))
        in->connect(c->output());
}
