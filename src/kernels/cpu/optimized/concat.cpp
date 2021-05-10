/* Copyright 2019-2020 Canaan Inc.
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
#include <nncase/kernels/cpu/optimized/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::optimized;

namespace
{
template <class T>
result<void> concat_impl(gsl::span<const gsl::byte *const> inputs, T *output, const runtime_shape_t &out_shape,
    gsl::span<const runtime_shape_t> &in_strides, NNCASE_UNUSED const runtime_shape_t &out_strides, size_t axis, const runtime_shape_t &concat_dims, NNCASE_UNUSED kernel_context &context) noexcept
{
    runtime_shape_t in_shape(out_shape);
    size_t elemsize = sizeof(T);
    auto subsize = std::accumulate(in_shape.begin() + (axis + 1), in_shape.end(), 1, std::multiplies<int>());
    auto *out_ptr = output;
    if (axis == 0)
    {
        for (size_t n = 0; n < inputs.size(); ++n)
        {
            const auto size = concat_dims[n] * subsize;
            memcpy(out_ptr, inputs[n], size * elemsize);
            out_ptr += size;
        }
    }
    else if (axis == 1)
    {
        for (size_t height = 0; height < in_shape[0]; ++height)
        {
            for (size_t n = 0; n < inputs.size(); ++n)
            {
                const auto size = concat_dims[n] * subsize;
                const auto in_offset = height * in_strides[n][0];
                const auto *in_ptr = reinterpret_cast<const T *>(inputs[n]) + in_offset;
                memcpy(out_ptr, in_ptr, size * elemsize);
                out_ptr += size;
            }
        }
    }
   else if (axis == 2)
    {
        for (size_t channel = 0; channel < in_shape[0]; ++channel)
        {
            for (size_t height = 0; height < in_shape[1]; ++height)
            {
                for (size_t n = 0; n < inputs.size(); ++n)
                {
                    const auto size = concat_dims[n] * subsize;
                    const auto in_offset = channel * in_strides[n][0] + height * in_strides[n][1];
                    auto *in_ptr = reinterpret_cast<const T*>(inputs[n]) + in_offset;
                    memcpy(out_ptr, in_ptr, size * elemsize);
                    out_ptr += size;
                }
            }
        }
    }
    else if (axis == 3)
    {
        for (size_t batch = 0; batch < in_shape[0]; ++batch)
        {
            for (size_t channel = 0; channel < in_shape[1]; ++channel)
            {
                for (size_t height = 0; height < in_shape[2]; ++height)
                {
                    for (size_t n = 0; n < inputs.size(); ++n)
                    {
                        const auto size = concat_dims[n] * subsize;
                        const auto in_offset = batch * in_strides[n][0] + channel * in_strides[n][1] + height * in_strides[n][2];
                        auto *in_ptr = reinterpret_cast<const T *>(inputs[n]) + in_offset;
                        memcpy(out_ptr, in_ptr, size * elemsize);
                        out_ptr += size;
                    }
                }
            }
        }
    }
    return ok();
}
}

#define CONCAT_IMPL(size, type) \
    case size:                  \
        return concat_impl(inputs, reinterpret_cast<type *>(output), out_shape, in_strides, out_strides, axis, concat_dims, context)

result<void> optimized::concat(datatype_t type, gsl::span<const gsl::byte *const> inputs, gsl::byte *output, const runtime_shape_t &out_shape,
    gsl::span<const runtime_shape_t> in_strides, const runtime_shape_t &out_strides, size_t axis, const runtime_shape_t &concat_dims, kernel_context &context) noexcept
{
    switch (runtime::get_bytes(type))
    {
        CONCAT_IMPL(1, uint8_t);
        CONCAT_IMPL(2, uint16_t);
        CONCAT_IMPL(4, uint32_t);
        CONCAT_IMPL(8, uint64_t);
    default:
        return err(std::errc::not_supported);
    }
}
