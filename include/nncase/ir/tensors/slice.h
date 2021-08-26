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
#include "../op.h"
#include "nncase/runtime/datatypes.h"
#include "opcode.h"

namespace nncase::ir::math
{
/** @brief Binary operator node */
class NNCASE_API binary_node : public op_node
{
public:
    DEFINE_NODE_OPCODE(op_math_binary);

    binary_node(binary_op_t binary_op);

    /** @brief Get the binary opcode of the binary expression */
    binary_op_t binary_op() const noexcept { return binary_op_; }
    /** @brief Set the binary opcode of the binary expression */
    void binary_op(binary_op_t value) noexcept { binary_op_ = value; }

private:
    binary_op_t binary_op_;
};

using binary = expr_t<binary_node>;
}
