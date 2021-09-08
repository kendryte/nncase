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
#include "../call.h"

namespace nncase::ir::F {
NNCASE_API call unary(unary_op_t unary_op, expr input);
NNCASE_API call binary(binary_op_t binary_op, expr lhs, expr rhs);

#define DEFINE_UNARY_FUNC(name, unary_op)                                      \
    inline call name(expr input) { return F::unary(unary_op, input); }

DEFINE_UNARY_FUNC(abs, unary_abs)
DEFINE_UNARY_FUNC(ceil, unary_ceil)
DEFINE_UNARY_FUNC(cos, unary_cos)
DEFINE_UNARY_FUNC(exp, unary_exp)
DEFINE_UNARY_FUNC(floor, unary_floor)
DEFINE_UNARY_FUNC(log, unary_log)
DEFINE_UNARY_FUNC(neg, unary_neg)
DEFINE_UNARY_FUNC(round, unary_round)
DEFINE_UNARY_FUNC(rsqrt, unary_rsqrt)
DEFINE_UNARY_FUNC(sin, unary_sin)
DEFINE_UNARY_FUNC(sqrt, unary_sqrt)
DEFINE_UNARY_FUNC(square, unary_square)
DEFINE_UNARY_FUNC(tanh, unary_tanh)
DEFINE_UNARY_FUNC(bitwise_not, unary_bitwise_not)
DEFINE_UNARY_FUNC(logical_not, unary_logical_not)

#undef DEFINE_UNARY_FUNC

#define DEFINE_BINARY_FUNC(name, binary_op)                                    \
    inline call name(expr lhs, expr rhs) {                                     \
        return F::binary(binary_op, lhs, rhs);                                 \
    }

DEFINE_BINARY_FUNC(add, binary_add)
DEFINE_BINARY_FUNC(sub, binary_sub)
DEFINE_BINARY_FUNC(mul, binary_mul)
DEFINE_BINARY_FUNC(div, binary_div)
DEFINE_BINARY_FUNC(mod, binary_mod)
DEFINE_BINARY_FUNC(min, binary_min)
DEFINE_BINARY_FUNC(max, binary_max)
DEFINE_BINARY_FUNC(pow, binary_pow)
DEFINE_BINARY_FUNC(bitwise_and, binary_bitwise_and)
DEFINE_BINARY_FUNC(bitwise_or, binary_bitwise_or)
DEFINE_BINARY_FUNC(bitwise_xor, binary_bitwise_xor)
DEFINE_BINARY_FUNC(logical_and, binary_logical_and)
DEFINE_BINARY_FUNC(logical_or, binary_logical_or)
DEFINE_BINARY_FUNC(logical_xor, binary_logical_xor)

#undef DEFINE_BINARY_FUNC
} // namespace nncase::ir::F

namespace nncase::ir {
inline call operator-(expr input) { return F::neg(input); }
inline call operator~(expr input) { return F::bitwise_not(input); }
inline call operator!(expr input) { return F::logical_not(input); }

inline call operator+(expr lhs, expr rhs) { return F::add(lhs, rhs); }
inline call operator-(expr lhs, expr rhs) { return F::sub(lhs, rhs); }
inline call operator*(expr lhs, expr rhs) { return F::mul(lhs, rhs); }
inline call operator/(expr lhs, expr rhs) { return F::div(lhs, rhs); }
inline call operator%(expr lhs, expr rhs) { return F::mod(lhs, rhs); }
inline call operator&(expr lhs, expr rhs) { return F::bitwise_and(lhs, rhs); }
inline call operator|(expr lhs, expr rhs) { return F::bitwise_or(lhs, rhs); }
inline call operator^(expr lhs, expr rhs) { return F::bitwise_xor(lhs, rhs); }
inline call operator&&(expr lhs, expr rhs) { return F::logical_and(lhs, rhs); }
inline call operator||(expr lhs, expr rhs) { return F::logical_or(lhs, rhs); }
} // namespace nncase::ir
