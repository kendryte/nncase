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
#include <nncase/runtime/stackvm/op_reader.h>

BEGIN_NS_NNCASE_RT_MODULE(stackvm)

class stack_entry
{
public:
    stack_entry() = default;

    stack_entry(uint8_t v) noexcept
        : stack_entry((intptr_t)v)
    {
    }

    stack_entry(uint16_t v) noexcept
        : stack_entry((intptr_t)v)
    {
    }

    stack_entry(uint32_t v) noexcept
        : stack_entry((intptr_t)v)
    {
    }

    stack_entry(int8_t v) noexcept
        : stack_entry((intptr_t)v)
    {
    }

    stack_entry(int16_t v) noexcept
        : stack_entry((intptr_t)v)
    {
    }

    stack_entry(int32_t v) noexcept
        : stack_entry((intptr_t)v)
    {
    }

    stack_entry(uintptr_t v) noexcept
        : stack_entry((intptr_t)v)
    {
    }

    stack_entry(intptr_t v) noexcept
        : i_(v), is_real_(false)
    {
    }

    stack_entry(bfloat16 v) noexcept
        : stack_entry((float)v)
    {
    }

    stack_entry(half v) noexcept
        : stack_entry((float)v)
    {
    }

    stack_entry(float v) noexcept
        : r_(v), is_real_(true)
    {
    }

    bool is_real() const noexcept { return is_real_; }

    uint8_t as_u1() const noexcept { return (uint8_t)i_; }
    uint16_t as_u2() const noexcept { return (uint16_t)i_; }
    uint32_t as_u4() const noexcept { return (uint32_t)i_; }
    int8_t as_i1() const noexcept { return (int8_t)i_; }
    int16_t as_i2() const noexcept { return (int16_t)i_; }
    int32_t as_i4() const noexcept { return (int32_t)i_; }
    int64_t as_i8() const noexcept { return (int64_t)i_; }
    uintptr_t as_u() const noexcept { return (uintptr_t)i_; }
    intptr_t as_i() const noexcept { return i_; }

    bfloat16 as_br2() const noexcept { return bfloat16::round_to_bfloat16(r_); }
    float as_r4() const noexcept { return r_; }
    float as_r() const noexcept { return r_; }

private:
    union
    {
        intptr_t i_;
        float r_;
    };
    bool is_real_;
};

class evaluate_stack
{
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
