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
#include "ref_ops.h"
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

namespace {
template <class T, class TOp>
result<void> binary_impl(TOp &&op, const T *lhs, const T *rhs, T *output,
                         std::span<const size_t> lhs_shape,
                         std::span<const size_t> lhs_strides,
                         std::span<const size_t> rhs_shape,
                         std::span<const size_t> rhs_strides,
                         std::span<const size_t> out_shape,
                         std::span<const size_t> out_strides,
                         NNCASE_UNUSED kernel_context &context) noexcept {
    if (is_scalar(out_shape)) {
        output[0] = op(lhs[0], rhs[0]);
        return ok();
    }
    return apply(out_shape, [&](std::span<const size_t> index) -> result<void> {
        const auto lhs_index =
            kernels::detail::get_reduced_offset(index, lhs_shape);
        const auto rhs_index =
            kernels::detail::get_reduced_offset(index, rhs_shape);
        const auto a = lhs[offset(lhs_strides, lhs_index)];
        const auto b = rhs[offset(rhs_strides, rhs_index)];
        output[offset(out_strides, index)] = op(a, b);
        return ok();
    });
}

#define BINARY_IMPL_OP(op, funct)                                              \
    case binary_op_t::op:                                                      \
        return binary_impl(funct, lhs, rhs, output, lhs_shape, lhs_strides,    \
                           rhs_shape, rhs_strides, out_shape, out_strides,     \
                           context)

result<void> binary_impl(binary_op_t op, const bool *lhs, const bool *rhs,
                         bool *output, std::span<const size_t> lhs_shape,
                         std::span<const size_t> lhs_strides,
                         std::span<const size_t> rhs_shape,
                         std::span<const size_t> rhs_strides,
                         std::span<const size_t> out_shape,
                         std::span<const size_t> out_strides,
                         NNCASE_UNUSED kernel_context &context) noexcept {
    switch (op) {
        BINARY_IMPL_OP(logical_and, [](bool a, bool b) { return (a && b); });
        BINARY_IMPL_OP(logical_or, [](bool a, bool b) { return (a || b); });
        BINARY_IMPL_OP(logical_xor, [](bool a, bool b) { return (a ^ b); });
    default:
        return err(std::errc::not_supported);
    }
}

template <class T>
result<void> binary_impl(binary_op_t op, const T *lhs, const T *rhs, T *output,
                         std::span<const size_t> lhs_shape,
                         std::span<const size_t> lhs_strides,
                         std::span<const size_t> rhs_shape,
                         std::span<const size_t> rhs_strides,
                         std::span<const size_t> out_shape,
                         std::span<const size_t> out_strides,
                         NNCASE_UNUSED kernel_context &context) noexcept {
    switch (op) {
        BINARY_IMPL_OP(add, std::plus<T>());
        BINARY_IMPL_OP(sub, std::minus<T>());
        BINARY_IMPL_OP(mul, std::multiplies<T>());
        BINARY_IMPL_OP(div, std::divides<T>());
        BINARY_IMPL_OP(min, [](T a, T b) { return std::min(a, b); });
        BINARY_IMPL_OP(max, [](T a, T b) { return std::max(a, b); });
        BINARY_IMPL_OP(pow, [](T a, T b) { return std::pow(a, b); });
        BINARY_IMPL_OP(logical_and,
                       [](T a, T b) { return static_cast<T>(a && b); });
        BINARY_IMPL_OP(mod, [](T a, T b) { return std::fmod(a, b); });
    default:
        return err(std::errc::not_supported);
    }
}

#define BINARY_IMPL(_ty)                                                       \
    return binary_impl(op, IN_CAST(_ty, lhs), IN_CAST(_ty, rhs),               \
                       OUT_CAST(_ty, output), lhs_shape, lhs_strides,          \
                       rhs_shape, rhs_strides, out_shape, out_strides,         \
                       context);
} // namespace

result<void> nncase::kernels::stackvm::reference::binary(
    typecode_t typecode, binary_op_t op, const std::byte *lhs,
    const std::byte *rhs, std::byte *output, std::span<const size_t> lhs_shape,
    std::span<const size_t> lhs_strides, std::span<const size_t> rhs_shape,
    std::span<const size_t> rhs_strides, std::span<const size_t> out_shape,
    std::span<const size_t> out_strides,
    NNCASE_UNUSED kernel_context &context) noexcept {
    if (typecode == dt_boolean) {
        return binary_impl(op, IN_CAST(bool, lhs), IN_CAST(bool, rhs),
                           OUT_CAST(bool, output), lhs_shape, lhs_strides,
                           rhs_shape, rhs_strides, out_shape, out_strides,
                           context);
    }
    TYPE_SELECT(typecode, BINARY_IMPL);
}
