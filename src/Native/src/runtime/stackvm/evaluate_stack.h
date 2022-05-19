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
#include <nncase/object.h>
#include <nncase/runtime/stackvm/op_reader.h>
#include <variant>
#include <vector>

BEGIN_NS_NNCASE_RT_MODULE(stackvm)

class stack_entry {
  public:
    stack_entry() = default;

    stack_entry(uint8_t v) noexcept : stack_entry((intptr_t)v) {}

    stack_entry(uint16_t v) noexcept : stack_entry((intptr_t)v) {}

    stack_entry(uint32_t v) noexcept : stack_entry((intptr_t)v) {}

    stack_entry(int8_t v) noexcept : stack_entry((intptr_t)v) {}

    stack_entry(int16_t v) noexcept : stack_entry((intptr_t)v) {}

    stack_entry(int32_t v) noexcept : stack_entry((intptr_t)v) {}

    stack_entry(uintptr_t v) noexcept : stack_entry((intptr_t)v) {}

    stack_entry(intptr_t v) noexcept : value_(v) {}

    stack_entry(bfloat16 v) noexcept : stack_entry((float)v) {}

    stack_entry(half v) noexcept : stack_entry((float)v) {}

    stack_entry(float v) noexcept : value_(v) {}

    stack_entry(dims_t v) noexcept : value_(std::move(v)) {}

    template <class T,
              class = std::enable_if_t<std::is_convertible_v<T, object>>>
    stack_entry(T v) noexcept : value_(object(std::move(v))) {}

    bool is_i() const noexcept { return value_.index() == 0; }
    bool is_r() const noexcept { return value_.index() == 1; }
    bool is_shape() const noexcept { return value_.index() == 2; }
    bool is_object() const noexcept { return value_.index() == 3; }

    uint8_t as_u1() const noexcept { return (uint8_t)as_i(); }
    uint16_t as_u2() const noexcept { return (uint16_t)as_i(); }
    uint32_t as_u4() const noexcept { return (uint32_t)as_i(); }
    int8_t as_i1() const noexcept { return (int8_t)as_i(); }
    int16_t as_i2() const noexcept { return (int16_t)as_i(); }
    int32_t as_i4() const noexcept { return (int32_t)as_i(); }
    uintptr_t as_u() const noexcept { return (uintptr_t)as_i(); }
    intptr_t as_i() const noexcept { return std::get<intptr_t>(value_); }

    bfloat16 as_br2() const noexcept {
        return bfloat16::round_to_bfloat16(as_r());
    }
    float as_r4() const noexcept { return as_r(); }
    float as_r() const noexcept { return std::get<float>(value_); }

    const dims_t &as_shape() const noexcept { return std::get<dims_t>(value_); }
    const object &as_object() const noexcept {
        return std::get<object>(value_);
    }

  private:
    std::variant<intptr_t, float, dims_t, object> value_;
};

class evaluate_stack {
  public:
    evaluate_stack() noexcept;

    bool empty() const noexcept;
    bool full() const noexcept;
    result<stack_entry> peek() noexcept;
    result<stack_entry> pop() noexcept;
    result<void> push(stack_entry entry) noexcept;

  private:
    std::vector<stack_entry> entries_;
    size_t top_;
};

END_NS_NNCASE_RT_MODULE
