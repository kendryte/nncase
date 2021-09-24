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
#include <nncase/transforms/egraph.h>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

#define RETURN_IF_VISITED()                                                    \
    {                                                                          \
        auto it = graph_.nodes_.find(ex);                                      \
        if (it != graph_.nodes_.end())                                         \
            return &it->second;                                                \
    }

class egraph::converter : public expr_functor<enode *> {
  public:
    using expr_functor::visit;

    converter(egraph &graph) : graph_(graph) {}

    enode *visit(const function &ex) override {
        RETURN_IF_VISITED();
        auto params = ex->parameters() |
                      ranges::views::transform(
                          [this](const expr &arg) { return visit(arg); }) |
                      ranges::to<std::vector>();
        auto body = visit(ex->body());

        auto mynode = graph_.make_node(ex);
        for (auto &par : params)
            mynode->children().emplace_back(&par->ecls());
        mynode->children().emplace_back(&body->ecls());
        return mynode;
    }

    enode *visit(const var &ex) override {
        RETURN_IF_VISITED();
        return graph_.make_node(ex);
    }

    enode *visit(const op &ex) override {
        RETURN_IF_VISITED();
        return graph_.make_node(ex);
    }

    enode *visit(const constant &ex) override {
        RETURN_IF_VISITED();
        return graph_.make_node(ex);
    }

    enode *visit(const call &ex) override {
        RETURN_IF_VISITED();
        auto target = visit(ex->target());
        auto args = ex->arguments() |
                    ranges::views::transform(
                        [this](const expr &arg) { return visit(arg); }) |
                    ranges::to<std::vector>();

        auto mynode = graph_.make_node(ex);
        mynode->children().emplace_back(&target->ecls());
        for (auto &arg : args)
            mynode->children().emplace_back(&arg->ecls());
        return mynode;
    }

    enode *visit(const tuple &ex) override {
        RETURN_IF_VISITED();
        auto fields = ex->fields() |
                      ranges::views::transform(
                          [this](const expr &field) { return visit(field); }) |
                      ranges::to<std::vector>();

        auto mynode = graph_.make_node(ex);
        for (auto &field : fields)
            mynode->children().emplace_back(&field->ecls());
        return mynode;
    }

  private:
    egraph &graph_;
};

enode::enode(ir::expr ex, eclass &ecls) : expr_(std::move(ex)), ecls_(&ecls) {}

eclass *egraph::add(const expr &ex) {
    auto it = nodes_.find(ex);
    if (it != nodes_.end())
        return &it->second.ecls();
    auto node = converter(*this)(ex);
    return &node->ecls();
}

enode *egraph::make_node(const expr &ex) {
    auto &cls = classes_.emplace_back(eclass{});
    auto &node =
        nodes_
            .emplace(std::piecewise_construct, std::forward_as_tuple(ex),
                     std::forward_as_tuple(ex, cls))
            .first->second;
    cls.nodes().emplace_back(&node);
    return &node;
}
