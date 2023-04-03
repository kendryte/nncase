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
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/stackvm/op_reader.h>
#include <variant>
#include <vector>

BEGIN_NS_NNCASE_RT_MODULE(stackvm)

class stack_entry {
  private:
    enum class entry_type { I, R, O };

  public:
    constexpr stack_entry() : type_(entry_type::I), i_(0) {}

    constexpr stack_entry(uint8_t v) noexcept : stack_entry((intptr_t)v) {}
    constexpr stack_entry(uint16_t v) noexcept : stack_entry((intptr_t)v) {}
    constexpr stack_entry(uint32_t v) noexcept : stack_entry((intptr_t)v) {}
    constexpr stack_entry(int8_t v) noexcept : stack_entry((intptr_t)v) {}
    constexpr stack_entry(int16_t v) noexcept : stack_entry((intptr_t)v) {}
    constexpr stack_entry(int32_t v) noexcept : stack_entry((intptr_t)v) {}
    constexpr stack_entry(uintptr_t v) noexcept : stack_entry((intptr_t)v) {}

    stack_entry(bfloat16 v) noexcept : stack_entry((float)v) {}
    stack_entry(half v) noexcept : stack_entry((float)v) {}

    constexpr stack_entry(intptr_t v) noexcept : type_(entry_type::I), i_(v) {}
    constexpr stack_entry(float v) noexcept : type_(entry_type::R), r_(v) {}

    template <class T,
              class = std::enable_if_t<std::is_convertible_v<T, object>>>
    stack_entry(T v) noexcept : type_(entry_type::O), o_(std::move(v)) {}

    stack_entry(const stack_entry &other) noexcept
        : type_(other.type_), i_(other.i_) {
        if (type_ == entry_type::O)
            o_.dangerous_add_ref();
    }

    constexpr stack_entry(stack_entry &&other) noexcept
        : type_(other.type_), i_(other.i_) {
        other.i_ = 0;
    }

    ~stack_entry() { destroy(); }

    stack_entry &operator=(const stack_entry &other) noexcept {
        destroy();
        type_ = other.type_;
        i_ = other.i_;
        if (type_ == entry_type::O)
            o_.dangerous_add_ref();
        return *this;
    }

    stack_entry &operator=(stack_entry &&other) noexcept {
        destroy();
        type_ = other.type_;
        i_ = other.i_;
        other.i_ = 0;
        return *this;
    }

    constexpr bool is_i() const noexcept { return type_ == entry_type::I; }
    constexpr bool is_r() const noexcept { return type_ == entry_type::R; }
    constexpr bool is_object() const noexcept { return type_ == entry_type::O; }

    constexpr uint8_t as_u1() const noexcept { return (uint8_t)as_i(); }
    constexpr uint16_t as_u2() const noexcept { return (uint16_t)as_i(); }
    constexpr uint32_t as_u4() const noexcept { return (uint32_t)as_i(); }
    constexpr int8_t as_i1() const noexcept { return (int8_t)as_i(); }
    constexpr int16_t as_i2() const noexcept { return (int16_t)as_i(); }
    constexpr int32_t as_i4() const noexcept { return (int32_t)as_i(); }
    constexpr uintptr_t as_u() const noexcept { return (uintptr_t)as_i(); }
    constexpr intptr_t as_i() const noexcept { return i_; }

    bfloat16 as_br2() const noexcept {
        return bfloat16::round_to_bfloat16(as_r());
    }
    float as_r4() const noexcept { return as_r(); }
    float as_r() const noexcept { return r_; }

    constexpr const object &as_object() const &noexcept { return o_; }
    object as_object() &&noexcept { return std::move(o_); }

  private:
    void destroy() {
        if (type_ == entry_type::O)
            std::destroy_at(&o_);
    }

  private:
    entry_type type_;
    union {
        intptr_t i_;
        float r_;
        object o_;
    };
};

#define DEFINE_STACK_PUSH(type)                                                \
    void push(type value) noexcept {                                           \
        if (!full()) {                                                         \
            new (top_++) stack_entry(std::move(value));                        \
        } else {                                                               \
            enlarge();                                                         \
            new (top_++) stack_entry(std::move(value));                        \
        }                                                                      \
    }

class evaluate_stack {
  public:
    evaluate_stack() noexcept;
    ~evaluate_stack();

    bool empty() const noexcept { return top_ == entries_; }
    bool full() const noexcept { return top_ == end_; }

    stack_entry &peek() noexcept {
        dbg_check(!empty());
        return top_[-1];
    }

    stack_entry pop() noexcept {
        dbg_check(!empty());
        return std::move(*--top_);
    }

    // Ensure the stack entry is not O.
    template <class T,
              class = std::enable_if_t<!std::is_convertible_v<T, object>>>
    T pop_nonobject() noexcept {
        dbg_check(!empty());
        if constexpr (std::is_integral_v<T>) {
            return static_cast<T>((--top_)->as_i());
        } else {
            return static_cast<T>((--top_)->as_r());
        }
    }

    object pop_object() noexcept {
        dbg_check(!empty());
        return std::move(*--top_).as_object();
    }

    template <class... TArgs> void push(TArgs &&...args) noexcept {
        if (!full()) {
            new (top_++) stack_entry(std::forward<TArgs>(args)...);
        } else {
            enlarge();
            new (top_++) stack_entry(std::forward<TArgs>(args)...);
        }
    }

    void push(stack_entry entry) noexcept;

  private:
    void enlarge() noexcept;

  private:
    stack_entry *entries_;
    stack_entry *top_;
    stack_entry *end_;
};

END_NS_NNCASE_RT_MODULE
