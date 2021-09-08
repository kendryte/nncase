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
#include "type.h"

namespace nncase::ir {
/** @brief Constant node */
class NNCASE_API constant_node : public expr_node {
  public:
    DEFINE_OBJECT_KIND(expr_node, object_constant);

    constant_node(type value_type, std::vector<std::byte> data);

    constant_node(type value_type, std::span<const std::byte> data)
        : constant_node(std::move(value_type),
                        std::vector<std::byte>(data.begin(), data.end())) {}

    template <class T>
    constant_node(type value_type, std::span<const T> data)
        : constant_node(std::move(type), std::as_bytes(data)) {}

    template <class TScalar>
    constant_node(TScalar scalar)
        : constant_node(type_t(to_datatype<TScalar>(), shape_t{1}),
                        std::span<const TScalar>(&scalar, 1)) {}

    /** @brief Get the type of the constant expression */
    const type &value_type() const noexcept { return value_type_; }
    /** @brief Get the mutable type of the constant expression */
    type &value_type() noexcept { return value_type_; }

    /** @brief Get the data of the constant expression */
    std::span<const std::byte> data() const noexcept { return data_; }
    /** @brief Get the mutable data of the constant expression */
    std::span<std::byte> data() noexcept { return data_; }

  private:
    type value_type_;
    std::vector<std::byte> data_;
};

using constant = object_t<constant_node>;
} // namespace nncase::ir
