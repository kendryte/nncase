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
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/placeholders.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/annotate_runtime.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

annotate_runtime_transform::annotate_runtime_transform(module_type_t module_type)
    : module_type_(module_type)
{
}

void annotate_runtime_transform::process(transform_context &context)
{
    for (auto n : context.matched_nodes)
        n->module_type(module_type_);
}

bool annotate_runtime_of_tag_node_transform::on_try_match(node &node, transform_context &context)
{
    auto &opcode = node.runtime_opcode();
    if (opcode == op_uninitialized
        || opcode == op_constant
        || opcode == op_bitcast)
    {
        auto conns = node.output_at(0).connections();
        if (std::any_of(conns.begin(), conns.end(), [&](input_connector *in) { return in->owner().module_type() != node.module_type(); }))
        {
            context.matched_nodes.emplace_back(&node);
            return true;
        }
    }

    // TODO: op_ignore

    return false;
}

void annotate_runtime_of_tag_node_transform::process(transform_context &context)
{
    auto &node = *context.matched_nodes[0];
    auto &opcode = node.runtime_opcode();

    if (auto un = node_cast<uninitialized>(node))
    {
        for (auto conn : dup(node.output_at(0).connections()))
        {
            if (conn->owner().module_type() != node.module_type())
            {
                auto new_uninit = context.graph.emplace<uninitialized>(un->output().type(), un->output().shape());
                new_uninit->name(un->name());
                new_uninit->module_type(conn->owner().module_type());
                new_uninit->output().connect(*conn);
            }
        }
    }
    else if (auto cast = node_cast<bitcast>(node))
    {
        for (auto conn : dup(node.output_at(0).connections()))
        {
            if (conn->owner().module_type() != node.module_type())
            {
                auto new_cast = context.graph.emplace<bitcast>(cast->input().type(), cast->input().shape(), cast->output().type(), cast->output().shape());
                new_cast->name(cast->name());
                new_cast->module_type(conn->owner().module_type());
                new_cast->input().connect(*cast->input().connection());
                new_cast->output().connect(*conn);
            }
        }
    }
    else if (auto con = node_cast<constant>(node))
    {
        for (auto conn : dup(node.output_at(0).connections()))
        {
            if (conn->owner().module_type() != node.module_type())
            {
                auto new_con = context.graph.emplace<constant>(con->output().type(), con->output().shape(), con->data());
                new_con->name(con->name());
                new_con->module_type(conn->owner().module_type());
                new_con->output().connect(*conn);
            }
        }
    }
    else
    {
        throw std::runtime_error("Annotate tag op: " + std::string(opcode.name) + "is not supported");
    }
}
