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

void make_concat_no_action_pass::run_core(graph &graph, nncase::target &target, const run_pass_options &options)
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

void make_bitcast_no_action_pass::run_core(graph &graph, nncase::target &target, const run_pass_options &options)
{
    auto alias_visitor = make_relay_ir_visitor([&](node &node) {
        if (auto b = node_cast<bitcast>(node))
        {
            if (b->attributes() & node_attr_action)
            {
                auto &out = *b->input().connection();
                auto cp = graph.emplace<copy>(out.type(), out.shape());
                cp->name(out.owner().name() + "/copy");
                cp->input().connect(out);
                b->input().connect(cp->output());
                b->attributes(b->attributes() & ~node_attr_action);
            }
        }
    });
    alias_visitor.visit(graph);
}

void add_copy_to_output_pass::run_core(graph &graph, nncase::target &target, const run_pass_options &options)
{
    auto alias_visitor = make_relay_ir_visitor([&](node &node) {
        if (auto o = node_cast<output_node>(node))
        {
            auto &out = *o->input().connection();
            auto cp = graph.emplace<copy>(out.type(), out.shape());
            cp->name(out.owner().name() + "/copy");
            cp->output().memory_location(mem_output);
            cp->input().connect(out);
            o->input().connect(cp->output());
        }
    });
    alias_visitor.visit(graph);
}

void alias_bitcast_buffer_pass::run_core(graph &graph, nncase::target &target, const run_pass_options &options)
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

void alias_concat_buffer_pass::run_core(graph &graph, nncase::target &target, const run_pass_options &options)
{
    //auto &context = *options.schedule_context;
    //auto alias_visitor = make_relay_ir_visitor([&](node &node) {
    //    if (auto b = node_cast<concat>(node))
    //    {
    //        if (!(b->attributes() & node_attr_action))
    //        {
    //            auto &input = *b->input().connection();
    //            auto &in_buf = context.logical_buffers.at(b->input().connection());
    //            auto &out_buf = context.logical_buffers.at(&b->output());

    //            // input & rdata should remain locations
    //            if (in_buf.memory_location() == mem_input || in_buf.memory_location() == mem_rdata)
    //            {
    //                // owner is input, parent shape is bitcast's
    //                shape_t begin(b->output().shape().size(), 0);
    //                out_buf.parent() = { &in_buf, begin, b->output().shape() };
    //                out_buf.strides_shape() = b->output().shape();
    //            }
    //            else
    //            {
    //                assert(in_buf.memory_location() == mem_data);

    //                // owner transfered to output
    //                shape_t begin(b->output().shape().size(), 0);
    //                in_buf.parent() = { &out_buf, begin, b->output().shape() };
    //                in_buf.strides_shape() = input.shape();
    //            }
    //        }
    //    }
    //});
    //alias_visitor.visit(graph);
}