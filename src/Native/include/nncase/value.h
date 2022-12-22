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
#include "object.h"
#include <vector>

namespace nncase {

class value_node;

/** @brief Value */
using value_t = object_t<value_node>;

class NNCASE_API value_node : public object_node {
    DEFINE_OBJECT_KIND(object_node, object_value);

  public:
    virtual result<void> copy_to(value_t dest) const noexcept = 0;
};

class NNCASE_API tuple_node : public value_node {
    DEFINE_OBJECT_KIND(value_node, object_tuple);

  public:
    tuple_node() noexcept = default;
    tuple_node(std::vector<value_t> fields) noexcept
        : fields_(std::move(fields)) {}

    gsl::span<const value_t> fields() const noexcept { return fields_; }
    gsl::span<value_t> fields() noexcept { return fields_; }

    result<void> copy_to(value_t dest) const noexcept override;

  private:
    std::vector<value_t> fields_;
};

/** @brief Tuple */
using tuple = object_t<tuple_node>;
} // namespace nncase
