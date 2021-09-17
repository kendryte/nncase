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
#include "function.h"

namespace nncase::ir {
/** @brief Module node*/
class NNCASE_API module_node : public object_node {
    DEFINE_OBJECT_KIND(object_node, object_module)
  public:
    module_node();

    /** @brief Get the functions of the module */
    const std::vector<function> &functions() const noexcept {
        return functions_;
    }

    /** @brief Add new funtion to the module */
    const function &add_function(function func);

    /** @brief Get the entry of the module */
    const function &entry() const noexcept { return entry_; }
    /** @brief Get the mutable entry of module */
    function &entry() noexcept { return entry_; }
    /** @brief Set the entry of the module */
    void entry(function value) noexcept { entry_ = std::move(value); }

  private:
    std::vector<function> functions_;
    function entry_;
};

/** @brief Module */
class NNCASE_API module_t : public object_t<module_node> {
  public:
    using object_t::object_t;
};
} // namespace nncase::ir
