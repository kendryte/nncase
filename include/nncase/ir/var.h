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
/** @brief Variable node */
class NNCASE_API var_node : public expr_node {
    DEFINE_OBJECT_KIND(expr_node, object_var)
  public:
    var_node(std::string name, type type_annotation = any_type()) noexcept;

    /** @brief Get the name of the variable expression */
    const std::string &name() const noexcept { return name_; }
    /** @brief Get the mutable name of the variable expression */
    std::string &name() noexcept { return name_; }

    /** @brief Get the type annotation of the variable expression */
    const type &type_annotation() const noexcept { return type_annotation_; }
    /** @brief Get the mutable type of the variable expression */
    type &type_annotation() noexcept { return type_annotation_; }

  private:
    std::string name_;
    type type_annotation_;
};

/** @brief Variable expression */
class var : public object_t<var_node> {
  public:
    using object_t::object_t;

    /** @brief Construct a variable expression with auto-generated name
     *  @param[in] type The type of the variable, default is any
     */
    NNCASE_API var(type type_annotation = any_type());

    /** @brief Construct a named variable expression
     *  @param[in] name The name of the variable
     *  @param[in] type The type of the variable, default is any
     */
    NNCASE_API var(std::string name, type type_annotation = any_type());
};
} // namespace nncase::ir
