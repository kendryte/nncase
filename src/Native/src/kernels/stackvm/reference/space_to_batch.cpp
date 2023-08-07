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
#include "../shape_infer.h"
#include "ref_ops.h"
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

namespace {
inline dims_t get_transposed_shape(const dims_t &input_shape,
                                   const dims_t &perm) {
    dims_t new_shape(input_shape.size());
    for (size_t i = 0; i < new_shape.size(); i++)
        new_shape[i] = input_shape[perm[i]];
    return new_shape;
}

template <class T>
result<void>
space_to_batch_impl([[maybe_unused]] datatype_t dt,
                    [[maybe_unused]] const T *input, [[maybe_unused]] T *output,
                    [[maybe_unused]] const dims_t &in_shape,
                    [[maybe_unused]] const dims_t &block_shape,
                    [[maybe_unused]] const paddings_t &paddings,
                    [[maybe_unused]] const dims_t &in_strides,
                    [[maybe_unused]] const dims_t &out_strides,
                    [[maybe_unused]] kernel_context &context) noexcept {
    auto in_shape_size = in_shape.size();
    auto spatial_size = block_shape.size();
    auto remain_shape_size = in_shape.size() - spatial_size - 1;
    auto new_paddings = paddings_t((1 + spatial_size + remain_shape_size));
    for (size_t i = 0; i < spatial_size; ++i) {
        new_paddings[1 + i] = paddings[i];
    }
    auto pad_out_shape =
        kernels::stackvm::pad_infer_shape(in_shape, new_paddings);
    auto size = compute_size(pad_out_shape);
    auto pad_output = std::make_unique<float[]>(size);
    auto pad_out_strides = get_default_strides(pad_out_shape);
    int64_t pad_value = 0;

    try_(kernels::stackvm::reference::pad(
        dt, reinterpret_cast<const gsl::byte *>(input),
        reinterpret_cast<gsl::byte *>(pad_output.get()), in_shape, in_strides,
        pad_out_strides, new_paddings, pad_mode_t::constant,
        IN_BYTE_CAST(&pad_value), context));

    dims_t new_shape;
    new_shape.reserve(in_shape_size + spatial_size);
    new_shape.assign(pad_out_shape.begin(),
                     pad_out_shape.begin() + in_shape_size - spatial_size);

    dims_t perms(in_shape_size - spatial_size);
    perms.reserve(in_shape_size + spatial_size);
    std::iota(perms.begin(), perms.begin() + in_shape_size - spatial_size, 0);

    dims_t spatial_perms;
    spatial_perms.reserve(spatial_size);

    for (size_t i = 0; i < spatial_size; i++) {
        size_t idx = in_shape_size - spatial_size + i;
        perms.push_back(new_shape.size());
        new_shape.push_back(pad_out_shape[idx] / block_shape[i]);

        spatial_perms.push_back(new_shape.size());
        new_shape.push_back(block_shape[i]);
    }

    perms.insert(perms.begin(), spatial_perms.begin(), spatial_perms.end());

    auto tp_shape = get_transposed_shape(new_shape, perms);
    auto tp_stride = get_default_strides(tp_shape);
    try_(kernels::stackvm::reference::transpose(
        dt, reinterpret_cast<const gsl::byte *>(pad_output.get()),
        reinterpret_cast<gsl::byte *>(output), new_shape, perms,
        get_default_strides(new_shape), tp_stride, context));
    return ok();
}
} // namespace

#define SPACE_TO_BATCH_IMPL(size, type)                                        \
    case size:                                                                 \
        return space_to_batch_impl(dt, reinterpret_cast<const type *>(input),  \
                                   reinterpret_cast<type *>(output), in_shape, \
                                   block_shape, paddings, in_strides,          \
                                   out_strides, context)

result<void> nncase::kernels::stackvm::reference::space_to_batch(
    datatype_t dt, const gsl::byte *input, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> block_shape,
    const paddings_t &paddings, gsl::span<const size_t> in_strides,
    [[maybe_unused]] gsl::span<const size_t> out_shape,
    gsl::span<const size_t> out_strides,
    NNCASE_UNUSED kernel_context &context) {
    switch (runtime::get_bytes(dt)) {
        SPACE_TO_BATCH_IMPL(1, uint8_t);
        SPACE_TO_BATCH_IMPL(2, uint16_t);
        SPACE_TO_BATCH_IMPL(4, uint32_t);
        SPACE_TO_BATCH_IMPL(8, uint64_t);
    default:
        return err(std::errc::not_supported);
    }
}