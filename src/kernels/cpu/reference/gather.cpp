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
template <class T>
result<void> gather_impl(const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &out_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, int32_t *indices, const runtime_shape_t &indices_shape, int32_t axis, int32_t batch_dims,
    kernel_context &context) noexcept
{
    return apply(out_shape, [&](const runtime_shape_t &out_index) -> result<void> {
        // select batch
        // [out_index.begin(), out_index.begin() + axis]
        runtime_shape_t in_index(in_shape.size());
        size_t i_index = 0;
        for (; i_index < axis; ++i_index)
        {
            in_index[i_index] = out_index[i_index];
        }

        // which index to be used in indices
        runtime_shape_t indices_index(out_index.begin() + axis, out_index.begin() + axis + indices_shape.size());
        auto indices_offset = offset(get_default_strides(indices_shape), indices_index);
        assert(indices_offset < compute_size(indices_shape));
        // select sub block in dim axis
        in_index[i_index] = indices[indices_offset];
        ++i_index;

        // select position in sub block
        for (auto o_index = axis + indices_shape.size(); o_index < out_index.size(); ++o_index)
        {
            in_index[i_index] = out_index[o_index];
            ++i_index;
        }
        auto o_offset = offset(out_strides, out_index);
        auto i_offset = offset(in_strides, in_index);
        assert(o_offset < compute_size(out_shape));
        assert(i_offset < compute_size(in_shape));
        output[offset(out_strides, out_index)] = input[offset(in_strides, in_index)];
        return ok();
    });
}
}

#define GATHER_IMPL(size, type) \
    case size:                  \
        return gather_impl(reinterpret_cast<const type *>(input), reinterpret_cast<type *>(output), in_shape, out_shape, in_strides, out_strides, indices, indices_shape, axis, batch_dims, context);

result<void> reference::gather(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape, const runtime_shape_t &out_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, int32_t *indices, const runtime_shape_t &indices_shape, int32_t axis, int32_t batch_dims,
    kernel_context &context) noexcept
{
    switch (runtime::get_bytes(type))
    {
        GATHER_IMPL(1, uint8_t);
        GATHER_IMPL(2, uint16_t);
        GATHER_IMPL(4, uint32_t);
        GATHER_IMPL(8, uint64_t);
    default:
        return err(std::errc::not_supported);
    }
}
