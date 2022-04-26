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
#include "../runtime_function.h"

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const neg_op_t &op) noexcept
{
    try_var(value, stack_.pop());
    if (!value.is_real())
        return stack_.push(-value.as_i());
    else
        return stack_.push(-value.as_r());
}

#define BINARY_IMPL(op)                           \
    try_var(b, stack_.pop());                     \
    try_var(a, stack_.pop());                     \
    if (!a.is_real())                             \
        return stack_.push(a.as_i() op b.as_i()); \
    else                                          \
        return stack_.push(a.as_r() op b.as_r())

#define BINARY_U_IMPL(op)                         \
    try_var(b, stack_.pop());                     \
    try_var(a, stack_.pop());                     \
    if (!a.is_real())                             \
        return stack_.push(a.as_u() op b.as_u()); \
    else                                          \
        return stack_.push(a.as_r() op b.as_r())

#define BINARY_BIT_IMPL(op)   \
    try_var(b, stack_.pop()); \
    try_var(a, stack_.pop()); \
    return stack_.push(a.as_i() op b.as_i());

#define BINARY_BIT_U_IMPL(op) \
    try_var(b, stack_.pop()); \
    try_var(a, stack_.pop()); \
    return stack_.push(a.as_u() op b.as_u());

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const add_op_t &op) noexcept
{
    BINARY_IMPL(+);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const sub_op_t &op) noexcept
{
    BINARY_IMPL(-);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const mul_op_t &op) noexcept
{
    BINARY_IMPL(*);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const div_op_t &op) noexcept
{
    BINARY_IMPL(/);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const div_u_op_t &op) noexcept
{
    BINARY_U_IMPL(/);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const rem_op_t &op) noexcept
{
    BINARY_IMPL(/);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const rem_u_op_t &op) noexcept
{
    BINARY_U_IMPL(/);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const and_op_t &op) noexcept
{
    BINARY_BIT_U_IMPL(&);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const or_op_t &op) noexcept
{
    BINARY_BIT_U_IMPL(|);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const xor_op_t &op) noexcept
{
    BINARY_BIT_U_IMPL(^);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const not_op_t &op) noexcept
{
    try_var(value, stack_.pop());
    return stack_.push(~value.as_u());
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const shl_op_t &op) noexcept
{
    BINARY_BIT_U_IMPL(<<);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const shr_op_t &op) noexcept
{
    BINARY_BIT_IMPL(>>);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const shr_u_op_t &op) noexcept
{
    BINARY_BIT_U_IMPL(>>);
}

#define COMPARE_IMPL(op)                                  \
    try_var(b, stack_.pop());                             \
    try_var(a, stack_.pop());                             \
    if (!a.is_real())                                     \
        return stack_.push(a.as_i() op b.as_i() ? 1 : 0); \
    else                                                  \
        return stack_.push(a.as_r() op b.as_r() ? 1 : 0)

#define COMPARE_U_IMPL(op)                                \
    try_var(b, stack_.pop());                             \
    try_var(a, stack_.pop());                             \
    if (!a.is_real())                                     \
        return stack_.push(a.as_u() op b.as_u() ? 1 : 0); \
    else                                                  \
        return stack_.push(a.as_r() op b.as_r() ? 1 : 0)

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const clt_op_t &op) noexcept
{
    COMPARE_IMPL(<);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const clt_u_op_t &op) noexcept
{
    COMPARE_U_IMPL(<);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const cle_op_t &op) noexcept
{
    COMPARE_IMPL(<=);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const cle_u_op_t &op) noexcept
{
    COMPARE_U_IMPL(<=);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ceq_op_t &op) noexcept
{
    COMPARE_U_IMPL(==);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const cge_op_t &op) noexcept
{
    COMPARE_IMPL(>=);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const cge_u_op_t &op) noexcept
{
    COMPARE_U_IMPL(>=);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const cgt_op_t &op) noexcept
{
    COMPARE_IMPL(>);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const cgt_u_op_t &op) noexcept
{
    COMPARE_U_IMPL(>);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const cne_op_t &op) noexcept
{
    COMPARE_U_IMPL(!=);
}
