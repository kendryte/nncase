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
template <class T> struct identity {
    T operator()(const T &src) const noexcept { return src; }
};

template <class TReducer, class TPostProcess, class T>
result<void>
reduce_impl(TReducer &&reducer, TPostProcess &&post_process, T init_value,
            const T *input, T *output, std::span<const size_t> in_shape,
            std::span<const size_t> axis, std::span<const size_t> in_strides,
            std::span<const size_t> out_shape,
            std::span<const size_t> out_strides, bool keep_dims,
            NNCASE_UNUSED kernel_context &context) noexcept {
    try_(apply(out_shape, [&](std::span<const size_t> index) -> result<void> {
        output[offset(out_strides, index)] = init_value;
        return ok();
    }));
    try_(apply(in_shape, [&](std::span<const size_t> index) -> result<void> {
        const auto v = input[offset(in_strides, index)];
        const auto out_index =
            kernels::detail::get_reduced_offset(index, axis, keep_dims);
        auto &dest = output[offset(out_strides, out_index)];
        dest = reducer(dest, v);
        return ok();
    }));
    try_(apply(out_shape, [&](std::span<const size_t> index) -> result<void> {
        auto &dest = output[offset(out_strides, index)];
        dest = post_process(dest);
        return ok();
    }));
    return ok();
}
} // namespace

#define REDUCE_IMPL(_ty, op, reducer, post_process)                            \
    case op:                                                                   \
        return reduce_impl(reducer, post_process,                              \
                           SCALAR_CAST(_ty, init_value), IN_CAST(_ty, input),  \
                           OUT_CAST(_ty, output), in_shape, axis, in_strides,  \
                           out_shape, out_strides, keep_dims, context)

#define REDUCE_IMPL_NO_POST(_ty, op, reducer)                                  \
    case op:                                                                   \
        return reduce_impl(reducer, identity<_ty>(),                           \
                           *reinterpret_cast<const _ty *>(init_value),         \
                           IN_CAST(_ty, input), OUT_CAST(_ty, output),         \
                           in_shape, axis, in_strides, out_shape, out_strides, \
                           keep_dims, context)

template <typename T>
result<void>
reduce_prod(const T *input, T *output, std::span<const size_t> in_shape,
            std::span<const size_t> in_strides,
            std::span<const size_t> out_strides_origin,
            std::span<const size_t> axes, bool keep_dims) noexcept {
    auto out_shape =
        kernels::detail::get_reduced_shape(in_shape, axes, keep_dims);
    auto out_strides =
        out_strides_origin.size() == 0 ? dims_t{1} : dims_t(out_strides_origin);
    // init with init_value
    try_(kernels::stackvm::apply(
        out_shape, [&](std::span<const size_t> index) -> result<void> {
            output[offset(out_strides, index)] = 1;
            return ok();
        }));

    try_(apply(in_shape, [&](std::span<const size_t> index) -> result<void> {
        const auto src = input[offset(in_strides, index)];
        auto out_idx =
            offset(out_strides,
                   kernels::detail::get_reduced_offset(index, axes, keep_dims));
        auto &dst = output[out_idx];
        dst *= src;
        return ok();
    }));

    return ok();
}

template <>
result<void>
reduce_prod(const bool *input, bool *output, std::span<const size_t> in_shape,
            std::span<const size_t> in_strides,
            std::span<const size_t> out_strides_origin,
            std::span<const size_t> axes, bool keep_dims) noexcept {
    auto out_shape =
        kernels::detail::get_reduced_shape(in_shape, axes, keep_dims);
    auto out_strides =
        out_strides_origin.size() == 0 ? dims_t{1} : dims_t(out_strides_origin);
    // init with init_value
    try_(kernels::stackvm::apply(
        out_shape, [&](std::span<const size_t> index) -> result<void> {
            output[offset(out_strides, index)] = 1;
            return ok();
        }));

    try_(apply(in_shape, [&](std::span<const size_t> index) -> result<void> {
        const auto src = input[offset(in_strides, index)];
        auto out_idx =
            offset(out_strides,
                   kernels::detail::get_reduced_offset(index, axes, keep_dims));
        auto &dst = output[out_idx];
        dst &= src;
        return ok();
    }));

    return ok();
}

template NNCASE_API result<void> reduce_prod<float>(
    const float *input, float *output, std::span<const size_t> in_shape,
    std::span<const size_t> in_strides, std::span<const size_t> out_strides,
    std::span<const size_t> axis, bool keep_dims) noexcept;

#define REDUCE_FULL_IMPL(_ty)                                                  \
    {                                                                          \
        auto out_shape =                                                       \
            kernels::detail::get_reduced_shape(in_shape, axis, keep_dims);     \
        switch (op) {                                                          \
            REDUCE_IMPL(                                                       \
                _ty, reduce_op_t::mean, std::plus<_ty>(),                      \
                [block_size = (_ty)kernels::detail::get_reduce_block_size(     \
                     in_shape, axis)](_ty v) { return v / block_size; });      \
            REDUCE_IMPL_NO_POST(_ty, reduce_op_t::min,                         \
                                [](_ty a, _ty b) { return std::min(a, b); });  \
            REDUCE_IMPL_NO_POST(_ty, reduce_op_t::max,                         \
                                [](_ty a, _ty b) { return std::max(a, b); });  \
            REDUCE_IMPL_NO_POST(_ty, reduce_op_t::sum, std::plus<_ty>());      \
        case reduce_op_t::prod:                                                \
            return reduce_prod(reinterpret_cast<const _ty *>(input),           \
                               reinterpret_cast<_ty *>(output), in_shape,      \
                               in_strides, out_strides, axis, keep_dims);      \
        default:                                                               \
            return err(std::errc::not_supported);                              \
        }                                                                      \
    }

result<void> nncase::kernels::stackvm::reference::reduce(
    typecode_t typecode, reduce_op_t op, const std::byte *init_value,
    const std::byte *input, std::byte *output, std::span<const size_t> in_shape,
    std::span<const size_t> axis, std::span<const size_t> in_strides,
    std::span<const size_t> out_strides, bool keep_dims,
    kernel_context &context) noexcept {
    TYPE_SELECT(typecode, REDUCE_FULL_IMPL);
}