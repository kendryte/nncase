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
#include "opcode.h"

namespace nncase::ir
{
#define DEFINE_NODE_OPCODE(value)                                 \
    static constexpr opcode_t opcode() noexcept { return value; } \
    const opcode_t &runtime_opcode() const noexcept override { return value; }

/** @brief Operator node */
class NNCASE_API op_node : public expr_node
{
public:
    DEFINE_NODE_NODEKIND(node_op);

    /** @brief Get the parameters of the function expression */
    std::span<const connector_info> parameters() const noexcept { return parameters_; }

    /** @brief Get the opcode of the operator expression */
    virtual const opcode_t &runtime_opcode() const noexcept = 0;

protected:
    connector_info &add_parameter(std::string name);

private:
    std::vector<connector_info> parameters_;
};

using op = expr_t<op_node>;
}
