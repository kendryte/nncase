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

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

#define RETURN_IF_VISITED()                                                    \
    {                                                                          \
        auto it = memo_.find(ex.get());                                        \
        if (it != memo_.end())                                                 \
            return it->second;                                                 \
    }

class egraph::converter : public expr_functor<enode *> {
  public:
    converter(egraph &graph) : graph_(graph) {}

    enode *visit(const function &ex) override {
        RETURN_IF_VISITED();
        auto name = "%" + ex->name();
        // 1. Function signature
        {
            in_function_body_ = false;
            names_.emplace(ex.get(), name);
            ident() << name << " = fn (";
            size_t i = 0;
            for (auto &par : ex->parameters()) {
                os_ << visit(par);
                if (++i != ex->parameters().size())
                    os_ << ", ";
            }
            os_ << ")";
            if (!ex->checked_type().empty())
                os_ << ": " << visit_type(ex->checked_type());
            os_ << " {" << std::endl;
        }

        // 2. Function body
        {
            in_function_body_ = true;
            ident_level_++;
            auto result = visit(ex->body());
            ident() << result << std::endl;
            ident_level_--;
        }

        // 3. Function closing
        ident() << "}" << std::endl;
        return name;
    }

    std::string visit(const var &ex) override {
        RETURN_IF_VISITED();
        auto name = "%" + ex->name();
        names_.emplace(ex.get(), name);
        if (!ex->type_annotation().empty())
            name += ": " + visit_type(ex->type_annotation());
        return name;
    }

    std::string visit(const op &ex) override {
        if (auto u = ex.as<math::unary>()) {
            return unary_op_to_string((*u)->unary_op());
        }
        if (auto b = ex.as<math::binary>()) {
            return binary_op_to_string((*b)->binary_op());
        }
        return std::string(ex->runtime_kind().name);
    }

    std::string visit(const constant &ex) override {
        RETURN_IF_VISITED();
        std::stringstream ss;
        ss << "const(";
        if (!ex->checked_type().empty())
            ss << visit_type(ex->checked_type());
        ss << ")";
        auto name = ss.str();
        names_.emplace(ex.get(), name);
        return name;
    }

    std::string visit(const call &ex) override {
        RETURN_IF_VISITED();
        auto target = visit(ex->target());
        auto args = ex->arguments() |
                    ranges::views::transform(
                        [this](const expr &arg) { return visit(arg); }) |
                    ranges::to<std::vector>();
        auto &name = alloc_temp_var(ex);
        ident() << name << " = " << target << "(";
        size_t i = 0;
        for (auto &arg : args) {
            os_ << arg;
            if (++i != args.size())
                os_ << ", ";
        }
        os_ << ")";
        if (!ex->checked_type().empty())
            os_ << ": " << visit_type(ex->checked_type());
        os_ << std::endl;
        return name;
    }

    std::string visit(const tuple &ex) override {
        RETURN_IF_VISITED();
        auto fields = ex->fields() |
                      ranges::views::transform(
                          [this](const expr &field) { return visit(field); }) |
                      ranges::to<std::vector>();
        auto &name = alloc_temp_var(ex);
        ident() << name << " = (";
        size_t i = 0;
        for (auto &field : fields) {
            os_ << field;
            if (++i != fields.size())
                os_ << ", ";
        }
        os_ << ")";
        if (!ex->checked_type().empty())
            os_ << ": " << visit_type(ex->checked_type());
        os_ << std::endl;
        return name;
    }

  private:
    egraph &graph_;
    std::unordered_map<expr_node *, enode *> memo_;
};

enode::enode(ir::expr ex, eclass &ecls) : expr_(std::move(ex)), ecls_(&ecls) {}

eclass *egraph::add(const expr &ex) {
    auto it = nodes_.find(ex);
    if (it != nodes_.end())
        return &it->second.ecls();
    auto &cls = classes_.emplace_back(eclass{});
    auto &node =
        nodes_
            .emplace(std::piecewise_construct, std::forward_as_tuple(ex),
                     std::forward_as_tuple(ex, cls))
            .first->second;
    cls.nodes().emplace_back(&node);
    return &cls;
}

enode *egraph::make_node(const expr &ex) {
    auto it = nodes_.find(ex);
    if (it != nodes_.end())
        return &it->second;
    auto &cls = classes_.emplace_back(eclass{});
    auto &node =
        nodes_
            .emplace(std::piecewise_construct, std::forward_as_tuple(ex),
                     std::forward_as_tuple(ex, cls))
            .first->second;
    cls.nodes().emplace_back(&node);
    return &node;
}

void egraph::fill_node_children(enode &node, const expr &ex) {}
