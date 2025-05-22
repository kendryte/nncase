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
#include "nncase/runtime/simple_types.h"
#include "value.h"

namespace nncase {
class paddings_value_node;
using paddings_value = object_t<paddings_value_node>;

class NNCASE_API paddings_value_node : public value_node {
    DEFINE_OBJECT_KIND(value_node, object_paddings);

  public:
    paddings_value_node(paddings_t value) noexcept : value_(std::move(value)) {}

    /** @brief Gets value. */
    const paddings_t &value() const noexcept { return value_; }

    /** @brief Gets or sets value. */
    void value(paddings_t value) noexcept { value_ = value; }

    result<void> copy_to(value_t dest) const noexcept override;

  private:
    paddings_t value_;
};
} // namespace nncase
