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
#include <iostream>
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
runtime_shape_t get_padded_shape(const runtime_shape_t &in_shape, const runtime_paddings_t &paddings)
{
    runtime_shape_t out_shape(in_shape.size());
    for (size_t i = 0; i < in_shape.size(); i++)
        out_shape[i] = (size_t)((int32_t)in_shape[i] + paddings[i].sum() + (in_shape[i] - 1) * paddings[i].interior);
    return out_shape;
}

inline runtime_shape_t get_transposed_shape(const runtime_shape_t &input_shape, const runtime_shape_t &perm)
{
    runtime_shape_t new_shape(input_shape.size());
    for (size_t i = 0; i < new_shape.size(); i++)
        new_shape[i] = input_shape[perm[i]];
    return new_shape;
}

template <class T>
result<void> space_to_batch_impl(datatype_t dt, const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &block_shape,
    const runtime_paddings_t &paddings, const runtime_shape_t &in_strides, [[maybe_unused]] const runtime_shape_t &out_strides, kernel_context &context) noexcept
{
    auto in_shape_size = in_shape.size();
    auto spatial_size = block_shape.size();
    auto new_paddings = runtime_paddings_t(in_shape_size, { 0, 0, 0 });
    for (size_t i = 0; i < spatial_size; ++i)
    {
        new_paddings[in_shape_size - spatial_size + i] = paddings[i];
    }
    auto pad_out_shape = get_padded_shape(in_shape, new_paddings);
    auto pad_output = std::make_unique<float[]>(compute_size(pad_out_shape));
    auto pad_out_strides = get_default_strides(pad_out_shape);
    scalar pad_value(0);

    try_(reference::pad(dt, reinterpret_cast<const gsl::byte *>(input),
        reinterpret_cast<gsl::byte *>(pad_output.get()), in_shape, in_strides,
        pad_out_strides, new_paddings,
        pad_mode_t::pad_constant,
        pad_value, context));

    runtime_shape_t new_shape;
    new_shape.reserve(in_shape_size + spatial_size);
    new_shape.assign(pad_out_shape.begin(), pad_out_shape.begin() + in_shape_size - spatial_size);

    runtime_shape_t perms(in_shape_size - spatial_size);
    perms.reserve(in_shape_size + spatial_size);
    std::iota(perms.begin(), perms.begin() + in_shape_size - spatial_size, 0);

    runtime_shape_t spatial_perms;
    spatial_perms.reserve(spatial_size);

    for (size_t i = 0; i < spatial_size; i++)
    {
        size_t idx = in_shape_size - spatial_size + i;
        perms.push_back(new_shape.size());
        new_shape.push_back(pad_out_shape[idx] / block_shape[i]);

        spatial_perms.push_back(new_shape.size());
        new_shape.push_back(block_shape[i]);
    }

    perms.insert(perms.begin(), spatial_perms.begin(), spatial_perms.end());

    auto tp_shape = get_transposed_shape(new_shape, perms);
    auto tp_stride = get_default_strides(tp_shape);
    try_(reference::transpose(dt, reinterpret_cast<const gsl::byte *>(pad_output.get()), reinterpret_cast<gsl::byte *>(output), new_shape, perms, get_default_strides(new_shape), tp_stride, context));
    return ok();
}
}

#define SPACE_TO_BATCH_IMPL(size, type) \
    case size:                          \
        return space_to_batch_impl(dt, reinterpret_cast<const type *>(input), reinterpret_cast<type *>(output), in_shape, block_shape, crops, in_strides, out_strides, context)

result<void> reference::space_to_batch(datatype_t dt, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &block_shape, const runtime_paddings_t &crops, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, kernel_context &context) noexcept
{
    switch (runtime::get_bytes(dt))
    {
        SPACE_TO_BATCH_IMPL(1, uint8_t);
        SPACE_TO_BATCH_IMPL(2, uint16_t);
        SPACE_TO_BATCH_IMPL(4, uint32_t);
        SPACE_TO_BATCH_IMPL(8, uint64_t);
    default:
        return err(std::errc::not_supported);
    }
}
