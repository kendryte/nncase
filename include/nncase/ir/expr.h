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
#include "nncase/runtime/compiler_defs.h"
#include "node_kind.h"
#include "type.h"

namespace nncase::ir
{
#define DEFINE_NODE_NODEKIND(value)                              \
    static constexpr node_kind kind() noexcept { return value; } \
    const node_kind &runtime_kind() const noexcept override { return value; }

/** @brief Expression node **/
class NNCASE_API expr_node
{
public:
    constexpr expr_node() noexcept = default;
    expr_node(const expr_node &) = delete;
    expr_node &operator=(const expr_node &) = delete;
    virtual ~expr_node();

    /** @brief Get the static type of the expression **/
    virtual const type_t &type() = 0;

    /** @brief Get the opcode of the expression **/
    virtual const node_kind &runtime_kind() const noexcept = 0;
};

class NNCASE_API expr
{
public:
    using node_type = expr_node;

    constexpr expr() noexcept = default;

    /** @brief Construct empty expression **/
    constexpr expr(std::nullptr_t) noexcept { }

    expr(std::shared_ptr<expr_node> node) noexcept
        : node_(std::move(node))
    {
    }

    /** @brief Get the managed pointer to the expression node **/
    expr_node *get() const noexcept { return node_.get(); }
    expr_node *operator->() const noexcept { return get(); }

private:
    std::shared_ptr<expr_node> node_;
};

template <class T>
class expr_t : public expr
{
public:
    using node_type = T;

    constexpr expr_t() noexcept = default;

    /** @brief Construct empty expression **/
    constexpr expr_t(std::nullptr_t) noexcept { }

    expr_t(std::shared_ptr<T> node) noexcept
        : expr(std::move(node))
    {
    }

    template <class... TArgs>
    expr_t(TArgs &&...args)
        : expr(std::make_shared<T>(std::forward<TArgs>(args)...))
    {
    }

    /** @brief Get the managed pointer to the expression node **/
    T *get() const noexcept { return static_cast<T *>(expr::get()); }
    T *operator->() const noexcept { return get(); }
};
}
