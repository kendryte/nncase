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
#include "../object.h"
#include "nncase/runtime/compiler_defs.h"
#include "node_kind.h"
#include "type.h"

namespace nncase::ir {
#define DEFINE_NODE_NODEKIND(value)                                            \
    static constexpr node_kind kind() noexcept { return value; }               \
    const node_kind &runtime_kind() const noexcept override { return value; }

/** @brief Expression node */
class NNCASE_API expr_node {
  public:
    constexpr expr_node() noexcept = default;
    expr_node(const expr_node &) = delete;
    expr_node &operator=(const expr_node &) = delete;
    virtual ~expr_node();

    /** @brief Get the kind of the expression */
    virtual const node_kind &runtime_kind() const noexcept = 0;
};

class expr {
  public:
    constexpr expr() = default;

    /** @brief Construct an empty expression **/
    constexpr expr(std::nullptr_t) noexcept : node_(nullptr) {}

    expr(std::shared_ptr<expr_node> node) noexcept : node_(std::move(node)) {}

    /** @brief Get the managed pointer to the object **/
    expr_node *get() const noexcept { return node_.get(); }
    expr_node *operator->() const noexcept { return get(); }

    /** @brief Is the expression an instance of specific type **/
    template <class T,
              class = std::enable_if_t<std::is_base_of_v<expr_node, T> ||
                                       std::is_base_of_v<expr, T>>>
    bool is_a() const noexcept {
        if constexpr (std::is_base_of_v<expr_node, T>)
            return node_ && node_->runtime_kind() == T::kind();
        else
            return node_ && node_->runtime_kind() == T::node_type::kind();
    }

  private:
    std::shared_ptr<expr_node> node_;
};

template <class T> class expr_t : public expr {
  public:
    static inline constexpr bool is_default_constructible_v =
        std::is_default_constructible_v<T> && !std::is_abstract_v<T>;

    using node_type = T;

    expr_t() {
        static_assert(is_default_constructible_v,
                      "Use nullptr to construct an empty expression.");
    }

    /** @brief Construct an empty expression **/
    constexpr expr_t(std::nullptr_t) noexcept : expr(nullptr) {}

    expr_t(std::shared_ptr<expr_node> node) noexcept : expr(std::move(node)) {}

    template <class = std::enable_if_t<!std::is_abstract_v<T>>, class... TArgs>
    expr_t(std::in_place_t, TArgs &&...args)
        : expr(std::make_shared<T>(std::forward<TArgs>(args)...)) {}

    /** @brief Get the managed pointer to the object **/
    T *get() const noexcept { return static_cast<T *>(expr::get()); }
    T *operator->() const noexcept { return get(); }

  private:
    std::shared_ptr<T> default_construct() {
        if constexpr (is_default_constructible_v)
            return std::make_shared<T>();
        else
            return nullptr;
    }
};
} // namespace nncase::ir
