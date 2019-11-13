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
#pragma once
#include "graph.h"
#include <unordered_set>

namespace nncase
{
namespace ir
{
    class ir_visitor
    {
    public:
        void visit(graph &graph);
        void visit(xtl::span<ir::output_node *> outputs);

        bool visited(node &node) const noexcept;

    protected:
        void mark_visit(node &node);

        virtual bool visit_strategry(node &node) = 0;

        virtual bool visit(node &node);

    private:
        std::unordered_set<node *> visited_;
    };

    class dfs_ir_visitor : public ir_visitor
    {
    protected:
        virtual bool visit_strategry(node &node) final override;

    private:
    };

    template <class TBaseVisitor, class TVisitor>
    class relay_ir_visitor : public TBaseVisitor
    {
    public:
        using TBaseVisitor::visit;

        relay_ir_visitor(TVisitor &&visitor)
            : visitor_(std::forward<TVisitor>(visitor))
        {
        }

    protected:
        virtual bool visit(node &node)
        {
            constexpr auto is_void = std::is_void_v<decltype(visitor_(node))>;
            if constexpr (is_void)
            {
                visitor_(node);
                return false;
            }
            else
            {
                return visitor_(node);
            }
        }

    private:
    private:
        TVisitor visitor_;
    };

    template <class TBaseVisitor = dfs_ir_visitor, class TVisitor>
    auto make_relay_ir_visitor(TVisitor &&visitor)
    {
        return relay_ir_visitor<TBaseVisitor, TVisitor>(std::forward<TVisitor>(visitor));
    }

    template <class TNode, class = std::enable_if_t<std::is_base_of_v<node, TNode>>>
    TNode *try_get_direct_child(node &node)
    {
        for (auto &&out : node.outputs())
        {
            for (auto &&conn : out.connections())
            {
                if (conn->owner().runtime_opcode() == TNode::opcode())
                    return static_cast<TNode *>(&conn->owner());
            }
        }

        return nullptr;
    }

    template <class TNode, class = std::enable_if_t<std::is_base_of_v<node, TNode>>>
    TNode *try_get_direct_parent(node &node)
    {
        for (auto &&in : node.inputs())
        {
            if (in.connection() && in.connection()->owner().runtime_opcode() == TNode::opcode())
                return static_cast<TNode *>(&in.connection()->owner());
        }

        return nullptr;
    }

    template <class TNode, class = std::enable_if_t<std::is_base_of_v<node, TNode>>>
    TNode *try_get_direct_parent(node &node, size_t index)
    {
        if (index < node.inputs().size())
        {
            auto &in = node.input_at(index);
            if (in.connection() && in.connection()->owner().runtime_opcode() == TNode::opcode())
                return static_cast<TNode *>(&in.connection()->owner());
        }

        return nullptr;
    }

    template <class TNode, class = std::enable_if_t<std::is_base_of_v<node, TNode>>>
    TNode *node_cast(node &node)
    {
        if (node.runtime_opcode() == TNode::opcode())
            return static_cast<TNode *>(&node);
        return nullptr;
    }
}
}
