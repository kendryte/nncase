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

namespace nncase::ir {
/** @brief Tuple node */
class NNCASE_API tuple_node : public expr_node {
    DEFINE_OBJECT_KIND(expr_node, object_tuple)
  public:
    tuple_node(std::vector<expr> fields);

    /** @brief Get the fields of the call expression */
    std::span<const expr> fields() const noexcept { return fields_; }
    /** @brief Get the mutable fields of the call expression */
    std::span<expr> fields() noexcept { return fields_; }

  private:
    std::vector<expr> fields_;
};

class tuple : public object_t<tuple_node> {
  public:
    using object_t::object_t;

    NNCASE_API tuple(std::vector<expr> fields);
};
} // namespace nncase::ir
