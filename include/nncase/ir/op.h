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
#include "ir_types.h"
#include "type_infer.h"
#include <utility>

namespace nncase::ir {
/** @brief Operator node */
class NNCASE_API op_node : public expr_node {
    DEFINE_OBJECT_KIND(expr_node, object_op)
  public:
    /** @brief Get the parameters of the function expression */
    std::span<const connector_info> parameters() const noexcept {
        return parameters_;
    }

    /** @brief Get the parameter at the index */
    const connector_info &parameter_at(size_t index) const noexcept {
        return parameters_.at(index);
    }

    /** @brief Infer the invoke result type */
    virtual type infer_invoke_result_type(type_infer_context &context) = 0;

  protected:
    connector_info &add_parameter(std::string name);

  private:
    std::vector<connector_info> parameters_;
};

using op = object_t<op_node>;
} // namespace nncase::ir
