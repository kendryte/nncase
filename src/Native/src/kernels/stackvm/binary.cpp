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
result<typecode_t> to_typecode(const datatype_t &dtype) {
    try_var(prim_type, dtype.as<prim_type_t>());
    return ok(prim_type->typecode());
}

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
        BINARY_IMPL_OP(pow, std::powf);
        BINARY_IMPL_OP(logical_and,
                       [](T a, T b) { return static_cast<T>(a && b); });
    default:
        return err(std::errc::not_supported);
    }
}

#define BINARY_IMPL_DTYPE(dtype, type)                                         \
    case dtype:                                                                \
        return binary_impl(op, reinterpret_cast<const type *>(lhs),            \
                           reinterpret_cast<const type *>(rhs),                \
                           reinterpret_cast<type *>(output), lhs_shape,        \
                           lhs_strides, rhs_shape, rhs_strides, out_shape,     \
                           out_strides, context);

result<void> binary_impl(typecode_t dtype, binary_op_t op, const gsl::byte *lhs,
                         const gsl::byte *rhs, gsl::byte *output,
                         const dims_t &lhs_shape, const strides_t &lhs_strides,
                         const dims_t &rhs_shape, const strides_t &rhs_strides,
                         const dims_t &out_shape, const strides_t &out_strides,
                         kernel_context &context) noexcept {
    switch (dtype) {
        BINARY_IMPL_DTYPE(dt_float32, float)
    default:
        return err(nncase_errc::datatype_mismatch);
    }
}
} // namespace

result<tensor> kernels::stackvm::binary(binary_op_t binary_op, tensor lhs,
                                        tensor rhs, tensor output,
                                        kernel_context &context) {
    auto dtype = lhs->dtype();
    try_var(typecode, to_typecode(dtype));
    auto out_shape =
        kernels::detail::get_binary_output_shape(lhs->shape(), rhs->shape());

    // TODO: copy back output
    assert(output.empty());
    if (output.empty()) {
        auto out_strides = get_default_strides(out_shape);
        try_var(out_buffer, buffer_allocator::host().allocate(
                                get_bytes(dtype, out_shape, out_strides), {}));
        output =
            tensor(std::in_place, dtype, out_shape, out_strides, out_buffer);
    } else {
        if (output->shape() != out_shape)
            return err(nncase_errc::shape_mismatch);
    }

    try_var(lhs_host, lhs->to_host());
    try_var(rhs_host, rhs->to_host());

    try_var(lhs_buffer, lhs_host->buffer().as_host());
    try_var(rhs_buffer, rhs_host->buffer().as_host());
    try_var(out_buffer, output->buffer().as_host());
    try_var(lhs_map, lhs_buffer.map(map_read));
    try_var(rhs_map, rhs_buffer.map(map_read));
    try_var(out_map, out_buffer.map(map_write));

    try_(binary_impl(typecode, binary_op, lhs_map.buffer().data(),
                     rhs_map.buffer().data(), out_map.buffer().data(),
                     lhs->shape(), lhs->strides(), rhs->shape(), rhs->strides(),
                     output->shape(), output->strides(), context));
    return ok(output);
}
