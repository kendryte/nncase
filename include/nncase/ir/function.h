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
#include "expr.h"
#include "var.h"

namespace nncase::ir
{
/** @brief Function node */
class NNCASE_API function_node : public expr_node
{
public:
    DEFINE_NODE_NODEKIND(node_function);

    function_node(std::vector<var> parameters, expr body);

    /** @brief Get the parameters of the function expression */
    std::span<const var> parameters() const noexcept { return parameters_; }

    /** @brief Get the body of the function expression */
    const expr &body() const noexcept { return body_; }
    /** @brief Set the body of the function expression */
    void body(expr value) noexcept { body_ = std::move(value); }

private:
    std::vector<var> parameters_;
    expr body_;
};

using function = expr_t<function_node>;
}
