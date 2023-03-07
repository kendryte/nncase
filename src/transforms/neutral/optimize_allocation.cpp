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
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/copy.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/visitor.h>
#include <nncase/schedule/scheduler.h>
#include <nncase/transforms/neutral/optimize_allocation.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;
using namespace nncase::schedule;

void make_concat_no_action_pass::run_core(graph &graph, [[maybe_unused]] nncase::target &target, [[maybe_unused]] const run_pass_options &options)
{
    auto alias_visitor = make_relay_ir_visitor([&](node &node) {
        if (auto c = node_cast<concat>(node))
        {
            if (c->attributes() & node_attr_action)
                c->attributes(c->attributes() & ~node_attr_action);
        }
    });
    alias_visitor.visit(graph);
}

void make_bitcast_no_action_pass::run_core(graph &graph, [[maybe_unused]] nncase::target &target, [[maybe_unused]] const run_pass_options &options)
{
    auto alias_visitor = make_relay_ir_visitor([&](node &node) {
        if (auto b = node_cast<bitcast>(node))
        {
            if (b->attributes() & node_attr_action)
            {
                auto &out = *b->input().connection();
                out.attributes(out.attributes() | cnctr_attr_no_layout_strides);
                b->output().attributes(out.attributes());
                b->attributes(b->attributes() & ~node_attr_action);
            }
        }
    });
    alias_visitor.visit(graph);
}

void make_slice_no_action_pass::run_core(graph &graph, [[maybe_unused]] nncase::target &target, [[maybe_unused]] const run_pass_options &options)
{
    auto alias_visitor = make_relay_ir_visitor([&](node &node) {
        if (auto s = node_cast<slice>(node))
        {
            if ((s->attributes() & node_attr_action)
                && is_copy_slice(s->strides()))
            {
                auto &out = s->output();
                out.attributes(out.attributes() | cnctr_attr_buffer_slice | cnctr_attr_no_buffer_fusion);
                s->attributes(s->attributes() & ~node_attr_action);
            }
        }
    });
    alias_visitor.visit(graph);
}

void add_copy_to_concat_pass::run_core(graph &graph, [[maybe_unused]] nncase::target &target, [[maybe_unused]] const run_pass_options &options)
{
    auto alias_visitor = make_relay_ir_visitor([&](node &node) {
        if (auto c = node_cast<concat>(node))
        {
            auto inputs = c->inputs();
            for (auto in : inputs)
            {
                auto &out = *in->connection();
                if (out.owner().runtime_opcode() != op_copy)
                {
                    auto cp = graph.emplace<copy>(out.type(), out.shape());
                    cp->name(out.owner().name() + "/copy");
                    cp->module_type(graph.module_type());
                    cp->input().connect(out);
                    in->connect(cp->output());
                }
            }
        }
    });
    alias_visitor.visit(graph);
}

void add_copy_to_slice_pass::run_core(graph &graph, [[maybe_unused]] nncase::target &target, [[maybe_unused]] const run_pass_options &options)
{
    auto alias_visitor = make_relay_ir_visitor([&](node &node) {
        slice *s;
        if ((s = node_cast<slice>(node))
            && (s->attributes() & node_attr_action) == 0
            && is_copy_slice(s->strides()))
        {
            auto outputs = dup(s->output().connections());
            if (std::any_of(outputs.begin(), outputs.end(), [](input_connector *in) { return in->owner().runtime_opcode() != op_copy; }))
            {
                auto cp = graph.emplace<copy>(s->output().type(), s->output().shape());
                cp->name(s->name() + "/copy");
                cp->module_type(graph.module_type());
                cp->input().connect(s->output());

                for (auto in : outputs)
                    in->connect(cp->output());
            }
        }
    });
    alias_visitor.visit(graph);
}

void add_copy_to_output_pass::run_core(graph &graph, [[maybe_unused]] nncase::target &target, [[maybe_unused]] const run_pass_options &options)
{
    auto alias_visitor = make_relay_ir_visitor([&](node &node) {
        if (auto o = node_cast<output_node>(node))
        {
            auto &out = *o->input().connection();
            if (out.owner().runtime_opcode() != op_copy
                || out.owner().output_at(0).connections().size() != 1)
            {
                auto cp = graph.emplace<copy>(out.type(), out.shape());
                cp->module_type(graph.module_type());
                cp->name(out.owner().name() + "/copy");
                cp->output().memory_location(mem_output);
                cp->input().connect(out);
                o->input().connect(cp->output());
            }
        }
    });
    alias_visitor.visit(graph);
}

void add_copy_to_bitcast_pass::run_core(graph &graph, [[maybe_unused]] nncase::target &target, [[maybe_unused]] const run_pass_options &options)
{
    auto alias_visitor = make_relay_ir_visitor([&](node &node) {
        if (auto b = node_cast<bitcast>(node))
        {
            auto &out = *b->input().connection();
            if (out.owner().runtime_opcode() != op_copy)
            {
                auto cp = graph.emplace<copy>(out.type(), out.shape());
                cp->module_type(graph.module_type());
                cp->name(out.owner().name() + "/copy");
                cp->input().connect(out);
                b->input().connect(cp->output());
            }
        }
    });
    alias_visitor.visit(graph);
}

//   x@data       x@output
//     |             |
//   copy            |
//     |     =>      |
//   output        output

bool remove_exclusive_copy_to_output_transform::on_try_match(node &node, transform_context &context)
{
    copy *cp;
    output_node *out;

    if ((cp = node_cast<copy>(node))
        && (out = try_get_direct_child<output_node>(*cp)))
    {
        auto input = cp->input().connection();
        if (input->memory_location() == mem_data
            && (input->attributes() & cnctr_attr_buffer_slice) == 0)
        {
            context.inputs.emplace_back(&cp->input());

            context.matched_nodes.emplace_back(cp);
            context.matched_nodes.emplace_back(out);
            return true;
        }
    }

    return false;
}

void remove_exclusive_copy_to_output_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto &old_out = static_cast<output_node &>(*context.matched_nodes[1]);
    if (output.connections().size() == 1)
        output.memory_location(mem_output);
    else
        output.memory_location(mem_shared_data);
    output.attributes(output.attributes() | cnctr_attr_no_layout_strides);
    old_out.input().connect(output);
}

//     x             x
//     |             |
//   copy            |
//     |     =>      |
//  concat        concat

bool remove_exclusive_copy_to_concat_transform::on_try_match(node &node, transform_context &context)
{
    copy *cp;
    concat *c, *pre_c;

    if ((cp = node_cast<copy>(node))
        && (c = try_get_direct_child<concat>(*cp)))
    {
        auto input = cp->input().connection();

        auto c_inputs = c->inputs();
        auto is_simple_concat = (c->axis() == 0 || std::all_of(c_inputs[0]->shape().begin(), c_inputs[0]->shape().begin() + c->axis(), [](size_t dim) { return dim == 1; }));
        if (input->memory_location() == mem_data
            && ((input->attributes() & (cnctr_attr_no_buffer_fusion | cnctr_attr_buffer_slice)) == 0)
            && (is_simple_concat || (input->attributes() & (cnctr_attr_no_layout_strides)) == 0))
        {
            if ((pre_c = try_get_direct_parent<concat>(*cp)) && pre_c->axis() != c->axis())
                return false;
            context.inputs.emplace_back(&cp->input());
            context.outputs.emplace_back(&cp->output());

            context.matched_nodes.emplace_back(cp);
            return true;
        }
    }

    return false;
}

void remove_exclusive_copy_to_concat_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    output.attributes(output.attributes() | cnctr_attr_no_buffer_fusion);
    for (auto &in : dup(inputs))
        in->connect(output);
}

bool remove_exclusive_copy_to_bitcast_transform::on_try_match(node &node, transform_context &context)
{
    copy *cp;
    bitcast *b;

    if ((cp = node_cast<copy>(node))
        && (b = try_get_direct_child<bitcast>(*cp)))
    {
        auto input = cp->input().connection();
        if ((input->memory_location() == mem_data || (input->memory_location() == mem_input && !try_get_direct_child<output_node>(*b)))
            && ((input->attributes() & cnctr_attr_no_buffer_fusion) == 0))
        {
            context.inputs.emplace_back(&cp->input());
            context.outputs.emplace_back(&cp->output());

            context.matched_nodes.emplace_back(cp);
            return true;
        }
    }

    return false;
}

void remove_exclusive_copy_to_bitcast_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    output.attributes(output.attributes() | cnctr_attr_no_buffer_fusion);
    for (auto &in : dup(inputs))
        in->connect(output);
}

//     x             x
//     |             |
//   slice           |
//     |     =>      |
//   copy          slice

bool remove_simple_copy_from_slice_transform::on_try_match(node &node, transform_context &context)
{
    slice *s;
    copy *cp;

    if ((s = node_cast<slice>(node))
        && ((s->attributes() & node_attr_action) == 0)
        && (cp = try_get_direct_child<copy>(*s))
        && !try_get_direct_child<output_node>(*cp)
        && !try_get_direct_child<concat>(*cp))
    {
        if (is_simple_slice(s->begin(), s->end(), s->strides(), s->input().shape()))
        {
            context.inputs.emplace_back(&cp->input());
            context.outputs.emplace_back(&cp->output());

            context.matched_nodes.emplace_back(cp);
            return true;
        }
    }

    return false;
}

void remove_simple_copy_from_slice_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    output.attributes(output.attributes() | cnctr_attr_no_layout_strides);
    for (auto &in : dup(inputs))
        in->connect(output);
}

bool remove_non_simple_copy_from_slice_transform::on_try_match(node &node, transform_context &context)
{
    slice *s;
    copy *cp;

    if ((s = node_cast<slice>(node))
        && ((s->attributes() & node_attr_action) == 0)
        && (cp = try_get_direct_child<copy>(*s))
        && !try_get_direct_child<output_node>(*cp)
        && !try_get_direct_child<concat>(*cp)
        && (cp->output().attributes() & cnctr_attr_no_layout_strides) == 0
        && (s->output().attributes() & cnctr_attr_no_layout_strides) == 0)
    {
        auto inputs = cp->output().connections();
        if (is_copy_slice(s->strides())
            && !is_simple_slice(s->begin(), s->end(), s->strides(), s->input().shape())
            && std::all_of(inputs.begin(), inputs.end(), [](input_connector *in) { return (in->attributes() & cnctr_attr_no_layout_strides) == 0; }))
        {
            context.inputs.emplace_back(&cp->input());
            context.outputs.emplace_back(&cp->output());

            context.matched_nodes.emplace_back(cp);
            return true;
        }
    }

    return false;
}

void remove_non_simple_copy_from_slice_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    for (auto &in : dup(inputs))
        in->connect(output);
}

void alias_bitcast_buffer_pass::run_core(graph &graph, [[maybe_unused]] nncase::target &target, const run_pass_options &options)
{
    auto &context = *options.schedule_context;
    auto alias_visitor = make_relay_ir_visitor([&](node &node) {
        if (auto b = node_cast<bitcast>(node))
        {
            if (!(b->attributes() & node_attr_action))
            {
                auto &input = *b->input().connection();
                auto &in_buf = *context.logical_buffer_map().at(b->input().connection());
                auto &out_buf = *context.logical_buffer_map().at(&b->output());

                size_t offset = 0;
                // input & rdata should remain locations
                if (in_buf.memory_location() == mem_input || in_buf.memory_location() == mem_rdata || in_buf.memory_location() == mem_output
                    || (input.attributes() & cnctr_attr_buffer_slice) || (in_buf.parent() && in_buf.parent()->parent != &out_buf))
                {
                    // owner is input, parent shape is bitcast's
                    out_buf.parent() = { &in_buf, offset, b->output().shape() };
                    out_buf.strides_parent() = is_axis0_squeeze_or_expand_dim_bitcast(in_buf.shape(), out_buf.shape())
                        ? &in_buf
                        : nullptr;
                }
                else
                {
                    assert(in_buf.memory_location() == mem_data);

                    // owner transfered to output
                    in_buf.parent() = { &out_buf, offset, b->output().shape() };
                    in_buf.strides_parent() = is_axis0_squeeze_or_expand_dim_bitcast(in_buf.shape(), out_buf.shape())
                        ? &out_buf
                        : nullptr;
                }
            }
        }
    });
    alias_visitor.visit(graph);
}

void alias_concat_buffer_pass::run_core([[maybe_unused]] graph &graph, [[maybe_unused]] nncase::target &target, [[maybe_unused]] const run_pass_options &options)
{
    auto &context = *options.schedule_context;

    auto alias_visitor = make_relay_ir_visitor([&](node &node) {
        if (auto c = node_cast<concat>(node))
        {
            if (c->attributes() & node_attr_action)
                return;

            auto axis = c->axis();
            auto &out_buf = context.logical_buffer_map().at(&c->output());
            shape_t cnt_begin(c->input_at(0).shape().size(), 0);
            size_t offset = 0;
            for (auto in : c->inputs())
            {
                auto &in_buf = context.logical_buffer_map().at(in->connection());
                in_buf->parent() = { out_buf, offset, c->output().shape() };
                in_buf->strides_parent() = out_buf;
                in_buf->memory_location() = out_buf->memory_location();
                cnt_begin[axis] += in->shape()[axis];
                offset = ir::get_bytes(in_buf->type()) * xt::element_offset<size_t>(to_strides(in_buf->parent()->shape), cnt_begin.begin(), cnt_begin.end());
            }
        }
    });
    alias_visitor.visit(graph);
}

void alias_slice_buffer_pass::run_core(graph &graph, [[maybe_unused]] nncase::target &target, const run_pass_options &options)
{
    auto &context = *options.schedule_context;
    auto alias_visitor = make_relay_ir_visitor([&](node &node) {
        if (auto s = node_cast<slice>(node))
        {
            if (!(s->attributes() & node_attr_action))
            {
                auto &in_buf = context.logical_buffer_map().at(s->input().connection());
                auto &out_buf = context.logical_buffer_map().at(&s->output());

                size_t offset = ir::get_bytes(in_buf->type()) * xt::element_offset<size_t>(to_strides(s->input().shape()), s->begin().begin(), s->begin().end());
                out_buf->parent() = { in_buf, offset, s->output().shape() };
                out_buf->strides_parent() = in_buf;
                out_buf->memory_location() = in_buf->memory_location();
            }
        }
    });
    alias_visitor.visit(graph);
}
