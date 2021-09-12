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
#include "../object.h"
#include "type.h"
#include <range/v3/range/concepts.hpp>

namespace nncase::ir {
/** @brief Expression node */
class NNCASE_API expr_node : public object_node {
  public:
    DEFINE_OBJECT_KIND(object_node, object_expr)

    /** @brief Get the checked type of the expression */
    const type &checked_type() const noexcept { return checked_type_; }
    /** @brief Get the mutable checked type of the variable expression */
    type &checked_type() noexcept { return checked_type_; }

  private:
    type checked_type_;
};

using expr = object_t<expr_node>;

template <class T>
concept Expr = Object<T> &&
    (concepts::same_as<expr_node, typename T::node_type> ||
     concepts::derived_from<typename T::node_type, expr_node>);
} // namespace nncase::ir
