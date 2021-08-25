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

namespace nncase::ir
{
/** @brief Variable node **/
class NNCASE_API var_node : public expr_node
{
public:
    DEFINE_NODE_NODEKIND(node_var);

    var_node(std::string name, type_t type) noexcept;

    const type_t &type() override;

    /** @brief Get the name of the variable expression **/
    const std::string &name() const noexcept { return name_; }
    /** @brief Get the mutable name of the variable expression **/
    std::string &name() noexcept { return name_; }

private:
    std::string name_;
    type_t type_;
};

using var = expr_t<var_node>;
}
