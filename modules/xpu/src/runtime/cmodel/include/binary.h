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

#include "../../gsl-lite.hpp"
#include <apply.h>
#include <runtime_utils.h>

using namespace nncase::runtime::xpu;

namespace kernels {

namespace {
template <class T, class TOp>
void binary_impl(TOp &&op, const T *lhs, const T *rhs, T *output,
                 gsl::span<const size_t> lhs_shape,
                 gsl::span<const size_t> lhs_strides,
                 gsl::span<const size_t> rhs_shape,
                 gsl::span<const size_t> rhs_strides,
                 gsl::span<const size_t> out_shape,
                 gsl::span<const size_t> out_strides) noexcept {
    if (is_scalar(out_shape)) {
        output[0] = op(lhs[0], rhs[0]);
        return;
    }
    return apply(out_shape, [&](gsl::span<const size_t> index) -> void {
        const auto lhs_index = get_reduced_offset(index, lhs_shape);
        const auto rhs_index = get_reduced_offset(index, rhs_shape);
        const auto a = lhs[offset(lhs_strides, lhs_index)];
        const auto b = rhs[offset(rhs_strides, rhs_index)];
        output[offset(out_strides, index)] = op(a, b);
        return;
    });
}

#define BINARY_IMPL_OP(op, funct)                                              \
    case binary_op_t::op:                                                      \
        return binary_impl(funct, lhs, rhs, output, lhs_shape, lhs_strides,    \
                           rhs_shape, rhs_strides, out_shape, out_strides)

template <class T>
void binary_impl(binary_op_t op, const T *lhs, const T *rhs, T *output,
                 gsl::span<const size_t> lhs_shape,
                 gsl::span<const size_t> lhs_strides,
                 gsl::span<const size_t> rhs_shape,
                 gsl::span<const size_t> rhs_strides,
                 gsl::span<const size_t> out_shape,
                 gsl::span<const size_t> out_strides) {
    // TODO: only support float for now
    if (std::is_same_v<T, float>) {
        switch (op) {
            BINARY_IMPL_OP(add, nncase_mt->float_binary_add);
            BINARY_IMPL_OP(sub, nncase_mt->float_binary_sub);
            BINARY_IMPL_OP(mul, nncase_mt->float_binary_mul);
            BINARY_IMPL_OP(div, nncase_mt->float_binary_div);
            BINARY_IMPL_OP(idenity_a,
                           [](T a, [[maybe_unused]] T b) { return a; });
            BINARY_IMPL_OP(min, nncase_mt->float_binary_min);
            BINARY_IMPL_OP(max, nncase_mt->float_binary_max);
            BINARY_IMPL_OP(pow, nncase_mt->float_binary_pow);
            BINARY_IMPL_OP(mod, nncase_mt->float_binary_mod);
            // BINARY_IMPL_OP(logical_and,
            //                [](T a, T b) { return static_cast<T>(a && b); });
            // BINARY_IMPL_OP(logical_or,
            //                [](T a, T b) { return static_cast<T>(a || b); });
            // BINARY_IMPL_OP(logical_xor,
            //                [](T a, T b) { return static_cast<T>(a ^ b); });
        default:
            runtime_util->rt_assert(false, (char *)"Unsupported Binary Op!");
        }
    } else if (std::is_same_v<T, int32_t>) {
        switch (op) {
            BINARY_IMPL_OP(add, nncase_mt->int32_binary_add);
            BINARY_IMPL_OP(sub, nncase_mt->int32_binary_sub);
            BINARY_IMPL_OP(mul, nncase_mt->int32_binary_mul);
            BINARY_IMPL_OP(div, nncase_mt->int32_binary_div);
            BINARY_IMPL_OP(idenity_a,
                           [](T a, [[maybe_unused]] T b) { return a; });
            BINARY_IMPL_OP(min, nncase_mt->int32_binary_min);
            BINARY_IMPL_OP(max, nncase_mt->int32_binary_max);
            BINARY_IMPL_OP(pow, nncase_mt->int32_binary_pow);
            BINARY_IMPL_OP(mod, nncase_mt->int32_binary_mod);
            // BINARY_IMPL_OP(logical_and,
            //                [](T a, T b) { return static_cast<T>(a && b); });
            // BINARY_IMPL_OP(logical_or,
            //                [](T a, T b) { return static_cast<T>(a || b); });
            // BINARY_IMPL_OP(logical_xor,
            //                [](T a, T b) { return static_cast<T>(a ^ b); });
        default:
            runtime_util->rt_assert(false, (char *)"Unsupported Binary Op!");
        }
    } else if (std::is_same_v<T, int64_t>) {
        switch (op) {
            BINARY_IMPL_OP(add, nncase_mt->int64_binary_add);
            BINARY_IMPL_OP(sub, nncase_mt->int64_binary_sub);
            BINARY_IMPL_OP(mul, nncase_mt->int64_binary_mul);
            BINARY_IMPL_OP(div, nncase_mt->int64_binary_div);
            BINARY_IMPL_OP(idenity_a,
                           [](T a, [[maybe_unused]] T b) { return a; });
            BINARY_IMPL_OP(min, nncase_mt->int64_binary_min);
            BINARY_IMPL_OP(max, nncase_mt->int64_binary_max);
            BINARY_IMPL_OP(pow, nncase_mt->int64_binary_pow);
            BINARY_IMPL_OP(mod, nncase_mt->int64_binary_mod);
            // BINARY_IMPL_OP(logical_and,
            //                [](T a, T b) { return static_cast<T>(a && b); });
            // BINARY_IMPL_OP(logical_or,
            //                [](T a, T b) { return static_cast<T>(a || b); });
            // BINARY_IMPL_OP(logical_xor,
            //                [](T a, T b) { return static_cast<T>(a ^ b); });
        default:
            runtime_util->rt_assert(false, (char *)"Unsupported Binary Op!");
        }
    } else {
        runtime_util->rt_assert(false, (char *)"Unsupported Binary Type!");
    }
}

} // namespace

template <class T>
void binary(binary_op_t op, const T *lhs, const T *rhs, T *output,
            gsl::span<const size_t> lhs_shape,
            gsl::span<const size_t> lhs_strides,
            gsl::span<const size_t> rhs_shape,
            gsl::span<const size_t> rhs_strides,
            gsl::span<const size_t> out_shape,
            gsl::span<const size_t> out_strides) noexcept {
    binary_impl(op, lhs, rhs, output, lhs_shape, lhs_strides, rhs_shape,
                rhs_strides, out_shape, out_strides);
}

} // namespace kernels