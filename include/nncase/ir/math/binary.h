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

namespace nncase::ir::math {
/** @brief Binary operator node */
class NNCASE_API binary_node : public op_node {
    DEFINE_OBJECT_KIND(op_node, op_math_binary)
  public:
    binary_node(binary_op_t binary_op);

    /** @brief Get the binary opcode of the binary expression */
    binary_op_t binary_op() const noexcept { return binary_op_; }
    /** @brief Set the binary opcode of the binary expression */
    void binary_op(binary_op_t value) noexcept { binary_op_ = value; }

    /** @brief Get the lhs the binary expression */
    const connector_info &lhs() const noexcept { return parameter_at(0); }
    /** @brief Get the rhs the binary expression */
    const connector_info &rhs() const noexcept { return parameter_at(1); }

    type infer_invoke_result_type(type_infer_context &context) override;

  private:
    binary_op_t binary_op_;
};

/** @brief Binary expression */
class binary : public object_t<binary_node> {
  public:
    using object_t::object_t;

    /** @brief Construct an binary expression
     *  @param[in] binary_op The opcode of the binary
     */
    NNCASE_API binary(binary_op_t binary_op);
};
} // namespace nncase::ir::math
