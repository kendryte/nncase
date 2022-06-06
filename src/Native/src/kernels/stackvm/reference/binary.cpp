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
#include <nncase/kernels/cpu/reference/runtime_types.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/util.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu::reference;
using namespace nncase::kernels::stackvm;

namespace {
template <class T, class TOp>
result<void> binary_impl(TOp &&op, const T *lhs, const T *rhs, T *output,
                         const dims_t &lhs_shape, const strides_t &lhs_strides,
                         const dims_t &rhs_shape, const strides_t &rhs_strides,
                         const dims_t &out_shape, const strides_t &out_strides,
                         NNCASE_UNUSED kernel_context &context) noexcept {
    return apply(out_shape, [&](const dims_t &index) -> result<void> {
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

template <class T>
result<void> binary_impl(binary_op_t op, const T *lhs, const T *rhs, T *output,
                         const dims_t &lhs_shape, const strides_t &lhs_strides,
                         const dims_t &rhs_shape, const strides_t &rhs_strides,
                         const dims_t &out_shape, const strides_t &out_strides,
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
    default:
        return err(std::errc::not_supported);
    }
}
} // namespace

result<value_t> kernels::stackvm::binary(binary_op_t binary_op, value_t lhs,
                                        value_t rhs, value_t output,
                                        kernel_context &context) {
    try_input(lhs_mem, lhs);
    try_input(rhs_mem, rhs);
    auto out_shape = detail::get_binary_output_shape(lhs_tensor->shape(), rhs_tensor->shape());
    try_output(out_mem, output, lhs_tensor->dtype(), out_shape);
    try_(binary_impl(binary_op, lhs_mem, rhs_mem, out_mem,
                     lhs_tensor->shape(), lhs_tensor->strides(),
                     rhs_tensor->shape(), rhs_tensor->strides(),
                     output_tensor->shape(), output_tensor->strides(), context));
    return ok(output);
}
