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
#include <nncase/kernels/apply.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

namespace {
template <class TOp, class T>
result<void> compare_impl(TOp &&op, const T *input_a, const T *input_b,
                          bool *output, const dims_t &in_a_shape,
                          const dims_t &in_a_strides, const dims_t &in_b_shape,
                          const dims_t &in_b_strides, const dims_t &out_shape,
                          const dims_t &out_strides) noexcept {
    return apply(out_shape, [&](const dims_t &index) -> result<void> {
        const auto in_a_index =
            kernels::detail::get_reduced_offset(index, in_a_shape);
        const auto in_b_index =
            kernels::detail::get_reduced_offset(index, in_b_shape);
        const auto a = input_a[offset(in_a_strides, in_a_index)];
        const auto b = input_b[offset(in_b_strides, in_b_index)];
        output[offset(out_strides, index)] = static_cast<bool>(op(a, b));
        return ok();
    });
}
} // namespace

#define COMPARE_IMPL_OP(op, funct)                                             \
    case compare_op_t::op:                                                     \
        return compare_impl(funct, lhs, rhs, output, lhs_shape, lhs_strides,   \
                            rhs_shape, rhs_strides, out_shape, out_strides)

template <typename T>
result<void> compare_impl(compare_op_t op, const T *lhs, const T *rhs,
                          bool *output, const dims_t &lhs_shape,
                          const dims_t &lhs_strides, const dims_t &rhs_shape,
                          const dims_t &rhs_strides, const dims_t &out_shape,
                          const dims_t &out_strides) noexcept {
    switch (op) {
        COMPARE_IMPL_OP(equal, std::equal_to<T>());
        COMPARE_IMPL_OP(not_equal, std::not_equal_to<T>());
        COMPARE_IMPL_OP(greater_than, std::greater<T>());
        COMPARE_IMPL_OP(greater_or_equal, std::greater_equal<T>());
        COMPARE_IMPL_OP(lower_than, std::less<T>());
        COMPARE_IMPL_OP(lower_or_equal, std::less_equal<T>());
    default:
        return err(std::errc::not_supported);
    }
}

#define COMPARE_IMPL(_ty)                                                      \
    return compare_impl(op, IN_CAST(_ty, lhs), IN_CAST(_ty, rhs),              \
                        OUT_CAST(bool, output), lhs_shape, lhs_strides,        \
                        rhs_shape, rhs_strides, out_shape, out_strides);

result<void> compare_impl(typecode_t typecode, compare_op_t op,
                          const gsl::byte *lhs, const gsl::byte *rhs,
                          gsl::byte *output, const dims_t &lhs_shape,
                          const strides_t &lhs_strides, const dims_t &rhs_shape,
                          const strides_t &rhs_strides, const dims_t &out_shape,
                          const strides_t &out_strides,
                          NNCASE_UNUSED kernel_context &context) noexcept {
    TYPE_SELECT(typecode, COMPARE_IMPL);
}

result<value_t>
kernels::stackvm::compare(compare_op_t compare_op, value_t lhs, value_t rhs,
                          value_t output,
                          [[maybe_unused]] kernel_context &context) {
    try_input(lhs_mem, lhs);
    try_input(rhs_mem, rhs);
    if (!cmp_dt(lhs_tensor, rhs_tensor)) {
        return err(nncase_errc::datatype_mismatch);
    }

    try_typecode(typecode, lhs_tensor);
    auto out_shape = nncase::kernels::detail::get_binary_output_shape(
        lhs_tensor->shape(), rhs_tensor->shape());
    try_output(out_mem, output, datatype_t::from_type<bool>(), out_shape);
    try_(compare_impl(
        typecode, compare_op, lhs_mem, rhs_mem, out_mem, lhs_tensor->shape(),
        lhs_tensor->strides(), rhs_tensor->shape(), rhs_tensor->strides(),
        output_tensor->shape(), output_tensor->strides(), context));
    return ok(output);
}
