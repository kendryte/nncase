/* Copyright 2019-2020 Canaan Inc.
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
#include "../runtime_module.h"

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

#define CONV_IMPL(type)                         \
    try_var(value, stack_.pop());               \
    if (!value.is_real())                       \
        return stack_.push((type)value.as_i()); \
    else                                        \
        return stack_.push((type)value.as_r())

result<void> stackvm_runtime_module::visit(const conv_i1_op_t &op) noexcept
{
    CONV_IMPL(int8_t);
}

result<void> stackvm_runtime_module::visit(const conv_i2_op_t &op) noexcept
{
    CONV_IMPL(int16_t);
}

result<void> stackvm_runtime_module::visit(const conv_i4_op_t &op) noexcept
{
    CONV_IMPL(int32_t);
}

result<void> stackvm_runtime_module::visit(const conv_i_op_t &op) noexcept
{
    CONV_IMPL(intptr_t);
}

result<void> stackvm_runtime_module::visit(const conv_u1_op_t &op) noexcept
{
    CONV_IMPL(uint8_t);
}

result<void> stackvm_runtime_module::visit(const conv_u2_op_t &op) noexcept
{
    CONV_IMPL(uint16_t);
}

result<void> stackvm_runtime_module::visit(const conv_u4_op_t &op) noexcept
{
    CONV_IMPL(uint32_t);
}

result<void> stackvm_runtime_module::visit(const conv_u_op_t &op) noexcept
{
    CONV_IMPL(uintptr_t);
}

result<void> stackvm_runtime_module::visit(const conv_br2_op_t &op) noexcept
{
    CONV_IMPL(bfloat16);
}

result<void> stackvm_runtime_module::visit(const conv_r4_op_t &op) noexcept
{
    CONV_IMPL(float);
}
