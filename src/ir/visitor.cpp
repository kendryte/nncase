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
#include <nncase/ir/visitor.h>
#include <queue>

using namespace nncase;
using namespace nncase::ir;

void expr_visitor::visit(const expr &ex) {
    if (!mark_visited(ex))
        expr_functor::visit(ex);
}

void expr_visitor::visit(const var &ex) {
    if (!ex->type_annotation().empty())
        visit_type(ex->type_annotation());
}

void expr_visitor::visit([[maybe_unused]] const constant &ex) {}

void expr_visitor::visit(const function &ex) {
    for (auto &param : ex->parameters())
        visit(param);
    visit(ex->body());
}

void expr_visitor::visit(const call &ex) {
    visit(ex->target());
    for (auto &arg : ex->arguments())
        visit(arg);
}

void expr_visitor::visit(const tuple &ex) {
    for (auto &field : ex->fields())
        visit(field);
}

void expr_visitor::visit([[maybe_unused]] const op &ex) {}

void expr_visitor::visit_type([[maybe_unused]] const type &t) {}

size_t expr_visitor::mark_visited(const expr &ex) {
    auto it = visit_count_.find(ex.get());
    if (it == visit_count_.end()) {
        visit_count_.emplace(ex.get(), 1);
        return 0;
    } else {
        return it->second++;
    }
}
// void ir_visitor::visit(graph &graph)
//{
//    visit(graph.outputs());
//}
//
// void ir_visitor::visit(std::span<output_node *> outputs)
//{
//    visited_.clear();
//
//    for (auto &&out : outputs)
//    {
//        if (visit_strategy(*out))
//            return;
//    }
//}
//
// bool ir_visitor::visited(node &node) const noexcept
//{
//    return visited_.find(&node) != visited_.end();
//}
//
// void ir_visitor::mark_visit(node &node)
//{
//    visited_.emplace(&node);
//}
//
// bool ir_visitor::visit([[maybe_unused]] node &node)
//{
//    return false;
//}
//
// bool dfs_ir_pre_order_visitor::visit_strategy(node &node)
//{
//    if (!visited(node))
//    {
//        mark_visit(node);
//
//        if (visit(node))
//            return true;
//
//        for (auto in : node.inputs())
//        {
//            if (in->connection())
//            {
//                if (visit_strategy(in->connection()->owner()))
//                    return true;
//            }
//        }
//    }
//
//    return false;
//}
//
// bool dfs_ir_post_order_visitor::visit_strategy(node &node)
//{
//    if (!visited(node))
//    {
//        mark_visit(node);
//
//        for (auto in : node.inputs())
//        {
//            if (in->connection())
//            {
//                if (visit_strategy(in->connection()->owner()))
//                    return true;
//            }
//        }
//
//        if (visit(node))
//            return true;
//    }
//
//    return false;
//}
//
// bool bfs_ir_pre_order_visitor::visit_strategy(node &node)
//{
//    std::queue<nncase::ir::node *> nodes;
//    std::unordered_set<nncase::ir::node *> nodes_set;
//    nodes.push(&node);
//    nodes_set.emplace(&node);
//
//    while (!nodes.empty())
//    {
//        auto p = nodes.front();
//        nodes.pop();
//        mark_visit(*p);
//        visit(*p);
//
//        for (auto in : p->inputs())
//        {
//            if (in->connection())
//            {
//                auto &in_node = in->connection()->owner();
//                if (nodes_set.emplace(&in_node).second)
//                    nodes.push(&in_node);
//            }
//        }
//    }
//
//    return false;
//}
