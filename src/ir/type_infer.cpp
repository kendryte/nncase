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
#include <nncase/ir/call.h>
#include <nncase/ir/type_infer.h>
#include <nncase/ir/visitor.h>
#include <range/v3/algorithm/all_of.hpp>
#include <range/v3/view/transform.hpp>
#include <unordered_map>

using namespace nncase;
using namespace nncase::ir;

namespace {
class type_infer_context_impl : public type_infer_context {
  public:
    type_infer_context_impl(
        std::unordered_map<expr_node *, type> &checked_types)
        : checked_types_(checked_types) {}

    type argument_type(const connector_info &parameter) override {
        return checked_type(argument(parameter));
    }

    expr argument(const connector_info &parameter) override {
        return current_call()->arguments()[parameter.index()];
    }

    const call &current_call() const noexcept { return cnt_call_; }
    void current_call(call value) noexcept { cnt_call_ = std::move(value); }

    const type &checked_type(const expr &ex) const {
        return checked_types_.at(ex.get());
    }

    void checked_type(const expr &ex, type t) {
        ex->checked_type(t);
        checked_types_.emplace(ex.get(), std::move(t));
    }

  private:
    call cnt_call_ = nullptr;
    std::unordered_map<expr_node *, type> &checked_types_;
};

#define RETURN_IF_VISITED()                                                    \
    {                                                                          \
        auto it = checked_types_.find(ex.get());                               \
        if (it != checked_types_.end())                                        \
            return it->second;                                                 \
    }

class type_infer_visitor : public expr_functor<type> {
  public:
    using expr_functor::visit;

    type_infer_visitor() : context_(checked_types_) {}

    bool fully_inferred() const noexcept { return fully_inferred_; }

    type visit(const function &ex) override {
        RETURN_IF_VISITED();
        itlib::small_vector<type> par_types;
        par_types.reserve(ex->parameters().size());
        for (auto &par : ex->parameters())
            par_types.emplace_back(visit(par));
        callable_type my_type(std::move(par_types), visit(ex->body()));
        checked_types_.emplace(ex.get(), my_type);
        ex->checked_type(my_type);
        return my_type;
    }

    type visit(const var &ex) override {
        RETURN_IF_VISITED();
        auto my_type = ex->type_annotation().value_or(any_type());
        checked_types_.emplace(ex.get(), my_type);
        ex->checked_type(my_type);
        if (my_type.is_a<any_type>())
            fully_inferred_ = false;
        return my_type;
    }

    type visit(const op &ex) override {
        RETURN_IF_VISITED();
        itlib::small_vector<type> par_types;
        par_types.reserve(ex->parameters().size());
        for (auto &par : ex->parameters())
            par_types.emplace_back(any_type());
        callable_type my_type(std::move(par_types), any_type());
        checked_types_.emplace(ex.get(), my_type);
        ex->checked_type(my_type);
        return my_type;
    }

    type visit(const constant &ex) override {
        RETURN_IF_VISITED();
        auto my_type = ex->value_type().value_or(any_type());
        checked_types_.emplace(ex.get(), my_type);
        ex->checked_type(my_type);
        if (my_type.is_a<any_type>())
            fully_inferred_ = false;
        return my_type;
    }

    type visit(const tuple &ex) override {
        RETURN_IF_VISITED();
        itlib::small_vector<type> field_types;
        field_types.reserve(ex->fields().size());
        for (auto &field : ex->fields())
            field_types.emplace_back(visit(field));
        tuple_type my_type(std::move(field_types));
        checked_types_.emplace(ex.get(), my_type);
        ex->checked_type(my_type);
        return my_type;
    }

    type visit(const call &ex) override {
        RETURN_IF_VISITED();
        auto &target = ex->target();
        auto target_type = visit(target);
        itlib::small_vector<type> args_types;
        args_types.reserve(ex->arguments().size());
        for (auto &arg : ex->arguments())
            args_types.emplace_back(visit(arg));

        context_.current_call(ex);
        type my_type(nullptr);
        if (auto func = target.as<function>()) {
            my_type = (*target_type.as<callable_type>())->return_type();
        } else {
            my_type = (*target.as<op>())->infer_invoke_result_type(context_);
        }

        checked_types_.emplace(ex.get(), my_type);
        ex->checked_type(my_type);
        if (my_type.is_a<any_type>())
            fully_inferred_ = false;
        return my_type;
    }

  private:
    bool fully_inferred_ = true;
    std::unordered_map<expr_node *, type> checked_types_;
    type_infer_context_impl context_;
};
} // namespace

bool ir::infer_type(const function &func) {
    type_infer_visitor v;
    v(func);
    return v.fully_inferred();
}

type ir::broadcast_type(std::initializer_list<tensor_type> inputs) {
    assert(inputs.size() >= 2);
    auto dtype = (*inputs.begin())->dtype();
    CHECK_TYPE(
        ranges::all_of(
            inputs, [=](const tensor_type &t) { return t->dtype() == dtype; }),
        "inputs must have same dtype");

    // If any input is not fixed, result is unranked
    if (ranges::any_of(inputs, [=](const tensor_type &t) {
            return !t->shape().is_fixed();
        }))
        return tensor_type(dtype, unranked_shape);

    shape_t out_shape(scalar_shape);
    const auto dest_rank =
        ranges::max(inputs | ranges::views::transform([](const tensor_type &t) {
                        return *t->shape().rank();
                    }));

    for (size_t dim_idx = 0; dim_idx < dest_rank; dim_idx++) {
        itlib::small_vector<size_t> in_dims;
        in_dims.reserve(inputs.size());
        for (auto it = inputs.begin(); it != inputs.end(); ++it) {
            const auto &in_shape = (*it)->shape();
            const auto in_extend = dest_rank - *in_shape.rank();
            auto in_dim = dim_value_t(dim_idx - in_extend);
            in_dim = in_dim < 0 ? 1 : in_shape[in_dim].fixed_value();
            assert(in_dim != 0);
            in_dims.emplace_back(in_dim);
        }

        // 1. Sort descending
        std::sort(in_dims.begin(), in_dims.end(), std::greater<>());
        assert(in_dims.front() > 0);
        // 2. Find first 1
        auto first_one = std::find(in_dims.begin(), in_dims.end(), 1);
        auto expected_dim = in_dims.front();
        // 3. Dims before 1 are all same or 1 is not found, it's ok to broadcast
        if (first_one == in_dims.end() ||
            std::all_of(in_dims.begin(), first_one,
                        [=](size_t dim) { return dim == expected_dim; })) {
            out_shape.emplace_back(expected_dim);
        } else {
            return invalid_type("inputs are not compatible to broadcast");
        }
    }

    return tensor_type(dtype, out_shape);
}
