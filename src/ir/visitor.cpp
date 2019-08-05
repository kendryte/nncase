/* Copyright 2019 Canaan Inc.
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
#include <ir/visitor.h>

using namespace nncase;
using namespace nncase::ir;

void ir_visitor::visit(graph &graph)
{
    visit(graph.outputs());
}

void ir_visitor::visit(xtl::span<ir::output_node *> outputs)
{
    visited_.clear();

    for (auto &&out : outputs)
    {
        if (visit_strategry(*out))
            return;
    }
}

bool ir_visitor::visited(node &node) const noexcept
{
    return visited_.find(&node) != visited_.end();
}

void ir_visitor::mark_visit(node &node)
{
    visited_.emplace(&node);
}

bool ir_visitor::visit(node &node)
{
    return false;
}

bool dfs_ir_visitor::visit_strategry(node &node)
{
    if (!visited(node))
    {
        mark_visit(node);

        for (auto &&in : node.inputs())
        {
            if (in.connection())
            {
                if (visit_strategry(in.connection()->owner()))
                    return true;
            }
        }

        if (visit(node))
            return true;

        for (auto &&out : node.outputs())
        {
            for (auto &&in : out.connections())
            {
                auto &owner = in->owner();
                if (owner.attributes() & node_attr_action)
                {
                    if (visit_strategry(owner))
                        return true;
                }
            }
        }
    }

    return false;
}
