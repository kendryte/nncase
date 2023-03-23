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

NNCASE_STACKVM_DISPATCH_BEGIN(NEG)
auto value = stack_.pop();
if (!value.is_r())
    stack_.push(-value.as_i());
else
    stack_.push(-value.as_r());
NNCASE_STACKVM_DISPATCH_END()

#define BINARY_IMPL(op)                                                        \
    auto b = stack_.pop();                                                     \
    auto a = stack_.pop();                                                     \
    if (!a.is_r())                                                             \
        stack_.push(a.as_i() op b.as_i());                                     \
    else                                                                       \
        stack_.push(a.as_r() op b.as_r())

#define BINARY_U_IMPL(op)                                                      \
    auto b = stack_.pop();                                                     \
    auto a = stack_.pop();                                                     \
    if (!a.is_r())                                                             \
        stack_.push(a.as_u() op b.as_u());                                     \
    else                                                                       \
        stack_.push(a.as_r() op b.as_r())

#define BINARY_BIT_IMPL(op)                                                    \
    auto b = stack_.pop();                                                     \
    auto a = stack_.pop();                                                     \
    stack_.push(a.as_i() op b.as_i())

#define BINARY_BIT_U_IMPL(op)                                                  \
    auto b = stack_.pop();                                                     \
    auto a = stack_.pop();                                                     \
    stack_.push(a.as_u() op b.as_u())

#define COMPARE_IMPL(op)                                                       \
    auto b = stack_.pop();                                                     \
    auto a = stack_.pop();                                                     \
    if (!a.is_r())                                                             \
        stack_.push(a.as_i() op b.as_i() ? 1 : 0);                             \
    else                                                                       \
        stack_.push(a.as_r() op b.as_r() ? 1 : 0)

#define COMPARE_U_IMPL(op)                                                     \
    auto b = stack_.pop();                                                     \
    auto a = stack_.pop();                                                     \
    if (!a.is_r())                                                             \
        stack_.push(a.as_u() op b.as_u() ? 1 : 0);                             \
    else                                                                       \
        stack_.push(a.as_r() op b.as_r() ? 1 : 0)

NNCASE_STACKVM_DISPATCH_BEGIN(ADD)
BINARY_IMPL(+);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(SUB)
BINARY_IMPL(-);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(MUL)
BINARY_IMPL(*);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(DIV)
BINARY_IMPL(/);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(DIV_U)
BINARY_U_IMPL(/);
NNCASE_STACKVM_DISPATCH_END()

#define NNCASE_STACKVM_DISPATCH_OP_REM                                         \
    auto b = stack_.pop();                                                     \
    auto a = stack_.pop();                                                     \
    if (!a.is_r())                                                             \
        stack_.push(a.as_i() % b.as_i());                                      \
    else                                                                       \
        stack_.push(fmodf(a.as_r(), b.as_r()));

#define NNCASE_STACKVM_DISPATCH_OP_REM_U                                       \
    auto b = stack_.pop();                                                     \
    auto a = stack_.pop();                                                     \
    if (!a.is_r())                                                             \
        stack_.push(a.as_u() % b.as_u());                                      \
    else                                                                       \
        stack_.push(fmodf(a.as_r(), b.as_r()));

NNCASE_STACKVM_DISPATCH_BEGIN(AND)
BINARY_BIT_U_IMPL(&);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(OR)
BINARY_BIT_U_IMPL(|);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(XOR)
BINARY_BIT_U_IMPL(^);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(NOT)
auto value = stack_.pop();
stack_.push(~value.as_u());
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(SHL)
BINARY_BIT_U_IMPL(<<);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(SHR)
BINARY_BIT_IMPL(>>);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(SHR_U)
BINARY_BIT_U_IMPL(>>);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CLT)
COMPARE_IMPL(<);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CLT_U)
COMPARE_U_IMPL(<);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CLE)
COMPARE_IMPL(<=);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CLE_U)
COMPARE_U_IMPL(<=);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CEQ)
COMPARE_U_IMPL(==);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CGE)
COMPARE_IMPL(>=);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CGE_U)
COMPARE_U_IMPL(>=);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CGT)
COMPARE_IMPL(>);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CGT_U)
COMPARE_U_IMPL(>);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CNE)
COMPARE_U_IMPL(!=);
NNCASE_STACKVM_DISPATCH_END()
