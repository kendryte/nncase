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

namespace nncase::ir::tensors {
/** @brief Cast operator node */
class NNCASE_API cast_node : public op_node {
    DEFINE_OBJECT_KIND(op_node, op_tensors_cast)
  public:
    cast_node(datatype_t new_type);

    /** @brief Get the new type of the cast expression */
    datatype_t new_type() const noexcept { return new_type_; }
    /** @brief Set the new type of the cast expression */
    void new_type(datatype_t value) noexcept { new_type_ = value; }

    /** @brief Get the input the unary expression */
    const connector_info &input() const noexcept { return parameter_at(0); }

    type infer_invoke_result_type(type_infer_context &context) override;

  private:
    datatype_t new_type_;
};

class cast : public object_t<cast_node> {
  public:
    NNCASE_API cast(datatype_t new_type);
};
} // namespace nncase::ir::tensors
