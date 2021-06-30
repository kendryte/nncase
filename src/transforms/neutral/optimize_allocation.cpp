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
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/copy.h>
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
            {
                auto inputs = c->inputs();
                for (auto in : inputs)
                {
                    auto &out = *in->connection();
                    auto cp = graph.emplace<copy>(out.type(), out.shape());
                    cp->name(out.owner().name() + "/copy");
                    cp->input().connect(out);
                    in->connect(cp->output());
                }

                c->attributes(c->attributes() & ~node_attr_action);
            }
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
                out.attributes(out.attributes() | cnctr_attr_no_buffer_fusion);
                b->attributes(b->attributes() & ~node_attr_action);
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
            if (out.owner().runtime_opcode() != op_copy)
            {
                auto cp = graph.emplace<copy>(out.type(), out.shape());
                cp->name(out.owner().name() + "/copy");
                cp->output().memory_location(mem_output);
                cp->input().connect(out);
                o->input().connect(cp->output());
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
        if (input->memory_location() == mem_data)
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

    output.memory_location(mem_output);
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
    concat *c;

    if ((cp = node_cast<copy>(node))
        && (c = try_get_direct_child<concat>(*cp)))
    {
        auto input = cp->input().connection();

        auto c_inputs = c->inputs();
        auto is_simple_concat = (c->axis() == 0 || std::all_of(c_inputs[0]->shape().begin(), c_inputs[0]->shape().begin() + c->axis(), [](size_t dim) { return dim == 1; }));
        if (input->memory_location() == mem_data
            && ((input->attributes() & cnctr_attr_no_buffer_fusion) == 0)
            && is_simple_concat)
        {
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

void alias_bitcast_buffer_pass::run_core(graph &graph, [[maybe_unused]] nncase::target &target, const run_pass_options &options)
{
    auto &context = *options.schedule_context;
    auto alias_visitor = make_relay_ir_visitor([&](node &node) {
        if (auto b = node_cast<bitcast>(node))
        {
            if (!(b->attributes() & node_attr_action))
            {
                auto &input = *b->input().connection();
                auto &in_buf = *context.logical_buffer_map.at(b->input().connection());
                auto &out_buf = *context.logical_buffer_map.at(&b->output());

                // input & rdata should remain locations
                if (in_buf.memory_location() == mem_input || in_buf.memory_location() == mem_rdata)
                {
                    // owner is input, parent shape is bitcast's
                    shape_t begin(b->output().shape().size(), 0);
                    out_buf.parent() = { &in_buf, begin, b->output().shape() };
                    out_buf.strides_shape() = b->output().shape();
                }
                else
                {
                    assert(in_buf.memory_location() == mem_data);

                    // owner transfered to output
                    shape_t begin(b->output().shape().size(), 0);
                    in_buf.parent() = { &out_buf, begin, b->output().shape() };
                    in_buf.strides_shape() = input.shape();
                }
            }
        }
    });
    alias_visitor.visit(graph);
}

//void alias_concat_buffer_pass::run_core([[maybe_unused]] graph &graph, [[maybe_unused]] nncase::target &target, [[maybe_unused]] const run_pass_options &options)
//{
//    auto &context = *options.schedule_context;
//    auto alias_visitor = make_relay_ir_visitor([&](node &node) {
//        if (auto c = node_cast<concat>(node))
//        {
//            if (!(c->attributes() & node_attr_action))
//            {
//                auto inputs = c->inputs();
//                auto outputs = c->output().connections();
//                auto is_simple_concat = (c->axis() == 0 || std::all_of(inputs[0]->shape().begin(), inputs[0]->shape().begin() + c->axis(), [](size_t dim) { return dim == 1; }));
//                auto &out_buf = *context.logical_buffer_map.at(&c->output());
//            }
//        }
//    });
//    alias_visitor.visit(graph);
//}
