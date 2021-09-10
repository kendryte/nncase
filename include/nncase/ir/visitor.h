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
#pragma once
#include "call.h"
#include "constant.h"
#include "expr.h"
#include "function.h"
#include "op.h"
#include "tuple.h"
#include "var.h"
#include <unordered_map>

namespace nncase::ir {

#define EXPR_FUNCTOR_DEFAULT                                                   \
    { return default_visit(ex); }

#define EXPR_FUNCTOR_DISPATCH(ex_type)                                         \
    if (auto n = ex.as<ex_type>())                                             \
    return visit(*n)

template <class R> class expr_functor {
  public:
    virtual ~expr_functor() = default;

    R operator()(const expr &ex) { return visit(ex); }

    virtual R default_visit(const expr &ex) {
        throw std::runtime_error("Undispatched expr: " +
                                 std::string(ex->runtime_kind().name));
    }

    virtual R visit(const expr &ex) {
        assert(!ex.empty());

        EXPR_FUNCTOR_DISPATCH(var);
        EXPR_FUNCTOR_DISPATCH(constant);
        EXPR_FUNCTOR_DISPATCH(function);
        EXPR_FUNCTOR_DISPATCH(call);
        EXPR_FUNCTOR_DISPATCH(tuple);
        EXPR_FUNCTOR_DISPATCH(op);
        return default_visit(ex);
    }

    virtual R visit(const var &ex) EXPR_FUNCTOR_DEFAULT;
    virtual R visit(const constant &ex) EXPR_FUNCTOR_DEFAULT;
    virtual R visit(const function &ex) EXPR_FUNCTOR_DEFAULT;
    virtual R visit(const call &ex) EXPR_FUNCTOR_DEFAULT;
    virtual R visit(const tuple &ex) EXPR_FUNCTOR_DEFAULT;
    virtual R visit(const op &ex) EXPR_FUNCTOR_DEFAULT;
};

#undef EXPR_FUNCTOR_DEFAULT
#undef EXPR_FUNCTOR_DISPATCH

class NNCASE_API expr_visitor : public expr_functor<void> {
  public:
    void visit(const expr &ex) override;
    void visit(const var &ex) override;
    void visit(const constant &ex) override;
    void visit(const function &ex) override;
    void visit(const call &ex) override;
    void visit(const tuple &ex) override;
    void visit(const op &ex) override;

    virtual void visit_type(const type &t);

  protected:
    size_t mark_visited(const expr &ex);

  private:
    std::unordered_map<expr_node *, size_t> visit_count_;
};
// class NNCASE_API dfs_ir_pre_order_visitor : public ir_visitor {
//  protected:
//    virtual bool visit_strategy(node &node) final override;
//
//  private:
//};
//
// class NNCASE_API dfs_ir_post_order_visitor : public ir_visitor {
//  protected:
//    virtual bool visit_strategy(node &node) final override;
//
//  private:
//};
//
// class NNCASE_API bfs_ir_pre_order_visitor : public ir_visitor {
//  protected:
//    virtual bool visit_strategy(node &node) final override;
//
//  private:
//};
//
// template <class TBaseVisitor, class TVisitor>
// class relay_ir_visitor : public TBaseVisitor {
//  public:
//    using TBaseVisitor::visit;
//
//    relay_ir_visitor(TVisitor &&visitor)
//        : visitor_(std::forward<TVisitor>(visitor)) {}
//
//  protected:
//    virtual bool visit(node &node) {
//        constexpr auto is_void = std::is_void_v<decltype(visitor_(node))>;
//        if constexpr (is_void) {
//            visitor_(node);
//            return false;
//        } else {
//            return visitor_(node);
//        }
//    }
//
//  private:
//  private:
//    TVisitor visitor_;
//};
//
// template <class TBaseVisitor = dfs_ir_post_order_visitor, class TVisitor>
// auto make_relay_ir_visitor(TVisitor &&visitor) {
//    return relay_ir_visitor<TBaseVisitor, TVisitor>(
//        std::forward<TVisitor>(visitor));
//}
//
// template <class TNode, class = std::enable_if_t<std::is_base_of_v<node,
// TNode>>> TNode *try_get_direct_child(node &node) {
//    for (auto out : node.outputs()) {
//        for (auto conn : out->connections()) {
//            if (conn->owner().runtime_opcode() == TNode::opcode())
//                return static_cast<TNode *>(&conn->owner());
//        }
//    }
//
//    return nullptr;
//}
//
// template <class TNode, class = std::enable_if_t<std::is_base_of_v<node,
// TNode>>> TNode *try_get_direct_parent(node &node) {
//    for (auto in : node.inputs()) {
//        if (in->connection() &&
//            in->connection()->owner().runtime_opcode() == TNode::opcode())
//            return static_cast<TNode *>(&in->connection()->owner());
//    }
//
//    return nullptr;
//}
//
// template <class TNode, class = std::enable_if_t<std::is_base_of_v<node,
// TNode>>> TNode *try_get_direct_parent(node &node, size_t index) {
//    if (index < node.inputs().size()) {
//        auto &in = node.input_at(index);
//        if (in.connection() &&
//            in.connection()->owner().runtime_opcode() == TNode::opcode())
//            return static_cast<TNode *>(&in.connection()->owner());
//    }
//
//    return nullptr;
//}
//
// template <class TNode, class = std::enable_if_t<std::is_base_of_v<node,
// TNode>>> TNode *node_cast(node &node) {
//    if (node.runtime_opcode() == TNode::opcode())
//        return static_cast<TNode *>(&node);
//    return nullptr;
//}
//
// inline size_t get_input_index(node &n, output_connector &input) {
//    for (size_t i = 0; i < n.inputs().size(); i++) {
//        if (n.input_at(i).connection() == &input)
//            return i;
//    }
//
//    throw std::out_of_range("Input connection not found");
//}
} // namespace nncase::ir
