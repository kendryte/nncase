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
#include "function.h"
#include "op.h"
#include <variant>

namespace nncase::ir
{
/** @brief Call node */
class NNCASE_API call_node : public expr_node
{
public:
    DEFINE_NODE_NODEKIND(node_call);

    call_node(std::variant<function, op> target, std::vector<expr> arguments);

    /** @brief Get the arguments of the call expression */
    std::span<const expr> arguments() const noexcept { return arguments_; }
    /** @brief Get the mutable arguments of the call expression */
    std::span<expr> arguments() noexcept { return arguments_; }

    /** @brief Get the target of the call expression */
    const std::variant<function, op> &target() const noexcept { return target_; }
    /** @brief Set the target of the function expression */
    void target(std::variant<function, op> value) noexcept { target_ = std::move(value); }

private:
    std::variant<function, op> target_;
    std::vector<expr> arguments_;
};

using call = expr_t<call_node>;
}
